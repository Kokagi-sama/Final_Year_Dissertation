# confusion_matrix.py
"""
Script for generating confusion matrices to evaluate ALPETNet model performance. 
(Can be used for LPETNet too though some configurations' changes must be made).

This script loads a pre-trained ALPETNet model, runs validation data through it,
and generates a confusion matrix to analyze prediction patterns and errors.
"""

import torch
import torch.nn as nn
from pathlib import Path
import random
import numpy as np
import gc
from torch.utils.data import DataLoader
import re

# Set multiprocessing start method to spawn for better CUDA compatibility
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from typing import Optional, Dict
from config.base_config import Config
from config.experiment_config import get_config
from models.ALPETNet import ALPETNet
from dataset_loader.lip_dataset import LipDataset
from trainer.trainer_comparative_analysis import ALPETNetTrainer

def set_random_seed(seed: int):
    """Set random seed for reproducibility across all random number generators.
    
    Sets seeds for Python's random module, NumPy, PyTorch CPU and GPU operations,
    and configures deterministic behavior.
    
    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)  # Set Python's random module seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU random seed
    torch.cuda.manual_seed_all(seed)  # Set PyTorch's GPU random seeds
    torch.backends.cudnn.deterministic = True  # Make operations deterministic
    torch.backends.cudnn.benchmark = False  # Disable autotuner for reproducibility

def build_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with appropriate settings.
    
    Args:
        config (Config): Configuration object containing dataset and
            training parameters.
    
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing (train_loader, val_loader).
    """
    val_dataset = LipDataset(config, phase='val')  # Initialize validation dataset

    # Enable pinned memory for faster data transfer to GPU if CUDA is available
    pin_memory = torch.cuda.is_available()
        
    # Create validation dataloader with no shuffling (order matters for evaluation)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False, # Don't shuffle validation data to maintain consistent order
        num_workers=config.training.num_workers,
        drop_last=False, # Keep all samples even if last batch is smaller
        pin_memory=pin_memory
    )
    
    return val_loader

def get_checkpoint(directory: Path) -> tuple[Path, float, float]:
    """Find the checkpoint in the given directory and extract CER and WER from filename.

    Args:
        directory (Path): Path to the checkpoint file.
        
    Returns:
        tuple: A tuple containing:
            - checkpoint_path (Path): Path to the checkpoint file
            - cer_value (float): Character Error Rate extracted from the filename
            - wer_value (float): Word Error Rate extracted from the filename
    """
    checkpoint_path = directory  # Use provided path directly
    cer_value = float("inf")  # Default to infinity if not found
    wer_value = float("inf")  # Default to infinity if not found

    # Define regex patterns to extract error rates from filename
    wer_pattern = re.compile(r"wer_([\d]+\.\d+)")
    cer_pattern = re.compile(r"cer_([\d]+\.\d+)")
    
    target_file = directory  # Use provided path as target file
    
    # Extract CER and WER from filename using regex
    cer_match = cer_pattern.search(target_file.name)
    wer_match = wer_pattern.search(target_file.name)
    
    if cer_match and wer_match:
        # Convert matched patterns to float values
        cer_value = float(cer_match.group(1))
        wer_value = float(wer_match.group(1))
        checkpoint_path = target_file
    else:
        print(f"Warning: Could not extract metrics from checkpoint filename: {target_file}")
        checkpoint_path = target_file
    
    return checkpoint_path, cer_value, wer_value


def load_model(config: Config) -> tuple[nn.Module, nn.Module, float, float, Optional[Dict], Optional[int]]:
    """Initialize and load model with checkpoint state if available.

    Args:
        config (Config): Configuration object containing model parameters
            and checkpoint paths.

    Returns:
        tuple: A tuple containing:
            - model (nn.Module): Base model instance
            - net (nn.Module): Parallelized model for multi-GPU training
            - cer (float): Latest Character Error Rate achieved
            - wer (float): Latest Word Error Rate achieved
            - optimizer_state_dict (Optional[Dict]): Saved optimizer state
            - ctr_tot (Optional[int]): Total training counter/iterations
            - train_losses (list): History of training losses
            - val_losses (list): History of validation losses
            - epochs_list (list): List of completed epochs
    """
    # Clear GPU memory before loading a new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Initialize the model with dropout parameters from config
    model = ALPETNet(
        dropout_p=config.model.cnn_dropout,
        transformer_dropout=config.model.trans_dropout
    )

    # Move model to GPU and create parallel version for multi-GPU training
    model = model.cuda()
    net = nn.DataParallel(model).cuda()
    
    # Default values if no checkpoint is found
    wer = float('inf')
    cer = float('inf')
    optimizer_state_dict = None
    ctr_tot = 0
    train_losses = []
    val_losses = []
    epochs_list = []

    # Load checkpoint if save path is provided
    if config.model.save_path:
        checkpoint, cer, wer = get_checkpoint(config.model.pretrained_path)
        
        if checkpoint: 
            # Load checkpoint using PyTorch's load function          
            checkpoint = torch.load(checkpoint, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Extract model weights
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                
                # Extract optimizer state if available
                if "optimizer_state_dict" in checkpoint:
                    optimizer_state_dict = checkpoint["optimizer_state_dict"]
                
                # Extract training counter if available
                if "ctr_tot" in checkpoint:
                    ctr_tot = checkpoint["ctr_tot"]
                
                # Use metrics from checkpoint data if available
                if "wer" in checkpoint:
                    wer = checkpoint["wer"]
                else:
                    wer = wer
                
                if "cer" in checkpoint:
                    cer = checkpoint["cer"]
                else:
                    cer = cer

                # Load training history
                if "train_losses" in checkpoint:
                    train_losses = checkpoint["train_losses"]

                if "val_losses" in checkpoint:
                    val_losses = checkpoint["val_losses"]

                if "epochs_list" in checkpoint:
                    epochs_list = checkpoint["epochs_list"]
            else:
                # Checkpoint is directly the state dict
                state_dict = checkpoint

            # Filter and load state dict, handling partial matches
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() 
                if k in model_dict and v.size() == model_dict[k].size()
            }

            # Check for any parameters that weren't loaded
            missed_params = [k for k in model_dict.keys() if k not in pretrained_dict]
            if missed_params:
                print('Missing parameters:', missed_params)

            # Update and load state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    return model, net, cer, wer, optimizer_state_dict, ctr_tot, train_losses, val_losses, epochs_list

def main():
    """Main training function that orchestrates the entire training pipeline.
    
    Loads configuration, initializes dataloaders and model, and executes
    the training loop with validation after each epoch. Handles checkpointing,
    metric tracking, error handling, and cleanup.
    """
    # Load configuration
    config = get_config()
  
    # Set random seed for reproducibility
    set_random_seed(config.model.random_seed)
    
    # Print GPU information
    if torch.cuda.is_available():
        print("CUDA Available:", torch.cuda.is_available())
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current CUDA Device:", torch.cuda.current_device())
        print("CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # Create validation dataloader
    val_loader = build_dataloaders(config)

    # Load model from checkpoint
    model, net, cer, wer, optimizer_state_dict, ctr_tot, train_losses, val_losses, epochs_list = load_model(config)
    
    # Initialize trainer with model and configuration
    trainer = ALPETNetTrainer(
        model=model,
        net=net,
        config=config,
        wer=wer,
        cer=cer,
        optimizer_state_dict=optimizer_state_dict,
        ctr_tot=ctr_tot
    )
    
    # Determine starting epoch (for single validation run)
    start_epoch = epochs_list[-1] if epochs_list else 0
    
    try:
        # Run only one validation epoch to generate confusion matrix
        for epoch in range(start_epoch, start_epoch + 1):          
            # Validate and generate confusion matrix
            val_loss, val_wer, val_cer, val_bleu, confusion_matrix_data = trainer.validate(val_loader, epoch)
           
            # Update metrics
            trainer.wer = val_wer
            trainer.cer = val_cer

            # Save confusion matrix data
            trainer.save_checkpoint(confusion_matrix_data)
                
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C)
        print("\nTraining interrupted by user")
    except Exception as e:
        # Handle other exception
        print(f"\nException occurred: {e}")

        # Try to save checkpoint before exiting
        if epochs_list:
            trainer.save_checkpoint()
        raise

if __name__ == '__main__':   
    # Run main function when script is executed directly
    main()