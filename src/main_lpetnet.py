# main_lpetnet.py
"""
Script for training and evaluating LPETNet lip reading models.

This script implements the full training pipeline for LPETNet models, including
data loading, training/validation cycles, checkpoint handling, and performance
monitoring. It tracks critical metrics like Character Error Rate (CER) and
Word Error Rate (WER) while ensuring proper resource management through
structured cleanup processes. The implementation supports multi-GPU training
and resuming from previous checkpoints.
"""
# Import necessary libraries for deep learning, file handling, and system operations
import torch
import torch.nn as nn
from pathlib import Path
import random
import numpy as np
import signal
import sys
import gc
import atexit
from torch.utils.data import DataLoader
import psutil
import os
import re

# Set multiprocessing start method to spawn for better CUDA compatibility
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

# Type hints and project-specific imports
from typing import Optional, Dict # Type annotations
from config.base_config import Config # Base configuration class
from config.experiment_config import get_config # Experiment configuration
from models.LPETNet import LPETNet # LPETNet model architecture
from dataset_loader.lip_dataset import LipDataset # Dataset for lip reading
from trainer.trainer_lpetnet import LPETNetTrainer # Trainer for LPETNet
 
# Global flag to prevent multiple cleanup calls
_is_cleaning_up = False

# Dictionary to track resources that need cleanup
resources = {}

def signal_handler(sig, frame):
    """Handle interrupt signals (Ctrl+C) by performing cleanup before exit.
    
    Args:
        sig: Signal number received from the operating system.
        frame: Current execution frame when the signal was received.
    """
    print("\nReceived SIGINT, cleaning up...")
    cleanup()
    sys.exit(0)

def cleanup_dataloader(dataloader):
    """Properly cleanup a DataLoader to prevent memory leaks.
    
    Handles releasing worker processes and removing dataset references
    to ensure proper garbage collection.
    
    Args:
        dataloader: The DataLoader instance to clean up.
    """
    if dataloader is None:
        return
    
    # Clear iterator to stop workers
    try:
        dataloader._iterator = None
    except:
        pass
    
    # Remove dataset reference to allow garbage collection
    try:
        dataloader.dataset = None
    except:
        pass
    
    # Force garbage collection
    gc.collect()

def safe_cleanup(resource_name):
    """Safely clean up a specific resource by name.
    
    Args:
        resource_name (str): Name of the resource to clean up, which should
            be a key in the global resources dictionary.
    """
    if resource_name in resources and resources[resource_name] is not None:
        print(f"Cleaning up {resource_name}...")
        
        # Special handling for dataloaders
        if resource_name in ['train_loader', 'val_loader']:
            cleanup_dataloader(resources[resource_name])
        
        # Clear the reference
        resources[resource_name] = None
    
    # First clean dataloaders
    safe_cleanup('train_loader')
    safe_cleanup('val_loader')
    
    # Then clean model objects
    safe_cleanup('trainer')
    safe_cleanup('net')
    safe_cleanup('model')

def cleanup():
    """Clean up all resources properly to avoid memory leaks and process zombies.
    
    Handles cleanup of dataloaders, models, CUDA resources, and child processes
    in the correct order to ensure proper resource release.
    """
    global _is_cleaning_up
    # Prevent recursive cleanup
    if _is_cleaning_up:
        return
    _is_cleaning_up = True
  
    # Clear resources dictionary
    resources.clear()
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # More aggressive cleanup for CUDA
        try:
            torch.cuda.ipc_collect()
        except:
            pass
            
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
    
    # Terminate any leftover child processes
    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        if children:
            print(f"Terminating {len(children)} child processes...")
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
    except:
        pass

    print("Cleanup completed.")

def set_random_seed(seed: int):
    """Set random seed for reproducibility across all random number generators.
    
    Sets seeds for Python's random module, NumPy, PyTorch CPU and GPU operations,
    and configures deterministic behavior.
    
    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)  # Python's random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Make operations deterministic
    torch.backends.cudnn.benchmark = False  # Disable autotuner

def build_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with appropriate settings.
    
    Args:
        config (Config): Configuration object containing dataset and
            training parameters.
    
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing (train_loader, val_loader).
    """
    # Initialize datasets for training and validation
    train_dataset = LipDataset(config, phase='train')
    val_dataset = LipDataset(config, phase='val')

    # Use pin_memory for faster GPU transfers if CUDA is available
    pin_memory = torch.cuda.is_available()
    
    # Create training dataloader with shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True, # Randomize order for training
        num_workers=config.training.num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    # Create validation dataloader without shuffling
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False, # Keep order for validation
        num_workers=config.training.num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def find_latest_checkpoint(directory: Path) -> tuple[Path, float, float]:
    """Find the latest checkpoint based on timestamp and extract CER and WER from filename.
    
    Searches for checkpoint files in the specified directory, sorts them by timestamp,
    and extracts performance metrics from the most recent checkpoint's filename.
    
    Args:
        directory (Path): Path to the directory containing checkpoint files.
    
    Returns:
        tuple[Path, float, float]: A tuple containing:
            - latest_checkpoint_path: Path to the most recent checkpoint file
            - latest_cer: Character Error Rate extracted from the filename
            - latest_wer: Word Error Rate extracted from the filename
    """
    # Convert to Path object if string was passed
    directory = Path(directory)

    # Verify directory exists
    if not directory.exists() or not directory.is_dir():
        print(f"Directory {directory} does not exist!")
        return None, float("inf"), float("inf")
    
    # Initialize variables with default values
    latest_checkpoint = None
    latest_cer = float("inf")  # Character Error Rate - lower is better
    latest_wer = float("inf") # Word Error Rate - lower is better

    # Define regex patterns for extracting information from filenames
    timestamp_pattern = re.compile(r"LPETNet_(\d{8}_\d{6})_")  # Format: YYYYMMDD_HHMMSS
    wer_pattern = re.compile(r"wer_([\d]+\.\d+)")  # WER pattern
    cer_pattern = re.compile(r"cer_([\d]+\.\d+)")  # CER pattern

    # Find all checkpoint files matching the pattern
    checkpoints = list(directory.glob("LPETNet_*.pt"))
    
    # Exit if no checkpoints found
    if not checkpoints:
        print("No checkpoints found!")
        return None, float("inf"), float("inf")
    
    # Sort checkpoints by timestamp in descending order (newest first)
    checkpoints.sort(key=lambda f: timestamp_pattern.search(f.name).group(1) if timestamp_pattern.search(f.name) else "", reverse=True)
    
    # Get the most recent checkpoint file (first in sorted list)
    latest_file = checkpoints[0]
    
    # Extract metrics from the filename using regex
    cer_match = cer_pattern.search(latest_file.name)
    wer_match = wer_pattern.search(latest_file.name)

    # If both metrics are found in the filename
    if cer_match and wer_match:
        # Extract and convert to float values
        latest_cer = float(cer_match.group(1))
        latest_wer = float(wer_match.group(1))
        latest_checkpoint = latest_file
        print(f"Selected latest checkpoint: {latest_checkpoint} with CER {latest_cer:.4f} and WER {latest_wer:.4f}")
    else:
         # Handle case where metrics aren't found in filename
        print(f"Warning: Could not extract metrics from latest checkpoint filename: {latest_file}")
        latest_checkpoint = latest_file

    # Return checkpoint path and metrics
    return latest_checkpoint, latest_cer, latest_wer

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
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Initialize the model with dropout parameters from config
    model = LPETNet(
        dropout_p=config.model.cnn_dropout,
        transformer_dropout=config.model.trans_dropout
    )

    # Move model to GPU and create parallel version for multi-GPU training
    model = model.cuda()
    net = nn.DataParallel(model).cuda()
    
    # Initialize training state variables
    wer = float('inf')
    cer = float('inf')
    optimizer_state_dict = None
    ctr_tot = 0
    train_losses = []
    val_losses = []
    epochs_list = []

    # Load checkpoint if save path is provided
    if config.model.save_path:
        latest_checkpoint, latest_cer, latest_wer = find_latest_checkpoint(config.model.save_path)
        
        if latest_checkpoint:
            print(f"Loading latest model from {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Extract model weights
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                
                # Extract training state if available
                if "optimizer_state_dict" in checkpoint:
                    optimizer_state_dict = checkpoint["optimizer_state_dict"]
                    print("Optimizer state loaded from checkpoint")
                
                if "ctr_tot" in checkpoint:
                    ctr_tot = checkpoint["ctr_tot"]
                    print(f"Training counter loaded: {ctr_tot}")
                
                # Extract performance metrics
                if "wer" in checkpoint:
                    wer = checkpoint["wer"]
                else:
                    wer = latest_wer
                
                if "cer" in checkpoint:
                    cer = checkpoint["cer"]
                else:
                    cer = latest_cer

                # Load training history
                if "train_losses" in checkpoint:
                    train_losses = checkpoint["train_losses"]

                if "val_losses" in checkpoint:
                    val_losses = checkpoint["val_losses"]

                if "epochs_list" in checkpoint:
                    epochs_list = checkpoint["epochs_list"]
                
                print(f"Using metrics from checkpoint: WER={wer:.4f}, CER={cer:.4f}")
            else:
                # Checkpoint is directly the state dict
                state_dict = checkpoint

            # Load weights into model, handling partial matches
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() 
                if k in model_dict and v.size() == model_dict[k].size()
            }
            
            # Report on loading statistics
            missed_params = [k for k in model_dict.keys() if k not in pretrained_dict]
            print(f'Loaded parameters: {len(pretrained_dict)}/{len(model_dict)}')
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
    
    # Create data loaders
    print("Creating dataloaders...")
    train_loader, val_loader = build_dataloaders(config)
    resources['train_loader'] = train_loader
    resources['val_loader'] = val_loader
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize or load model from checkpoint
    print("Loading model...")
    model, net, cer, wer, optimizer_state_dict, ctr_tot, train_losses, val_losses, epochs_list = load_model(config)
    
    # Create trainer with model and configuration
    trainer = LPETNetTrainer(
        model=model,
        net=net,
        config=config,
        wer=wer,
        cer=cer,
        optimizer_state_dict=optimizer_state_dict,
        ctr_tot=ctr_tot
    )
    
    # Store model objects for cleanup
    resources['model'] = model
    resources['net'] = net
    resources['trainer'] = trainer

    # Determine starting epoch (for resuming training)
    start_epoch = epochs_list[-1] if epochs_list else 0
    
    try:
        # Main training loop
        for epoch in range(start_epoch, config.training.max_epoch):
            print(f"\nEpoch {epoch+1}/{config.training.max_epoch}")
            
             # Training phase
            train_loss = trainer.train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_wer, val_cer, val_bleu = trainer.validate(val_loader, epoch)
            val_losses.append(val_loss)
            epochs_list.append(epoch + 1)
            
            # Log metrics
            print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}, BLEU: {val_bleu:.4f}")
            print(f"\nEpoch {epoch+1} - Val Loss: {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}, BLEU: {val_bleu:.4f}")
            
            # Update metrics
            trainer.wer = val_wer
            trainer.cer = val_cer

            # Save checkpoint
            print("Save checkpoint called!")
            trainer.save_checkpoint(val_loss, val_wer, val_cer, val_bleu, train_losses, val_losses, epochs_list)
            print(f"Saved latest model with WER: {val_wer:.4f}, CER: {val_cer:.4f} and BLEU: {val_bleu:.4f}")
            
            # Plot training progress
            trainer.plot_losses(train_losses, val_losses, epochs_list)
                
    except KeyboardInterrupt:
        # Handle user interruption
        print("\nTraining interrupted by user")
    except Exception as e:
        # Handle other exceptions
        print(f"\nException occurred: {e}")

        # Save progress before exiting
        if epochs_list:
            trainer.plot_losses(train_losses, val_losses, epochs_list)
            trainer.save_checkpoint(val_loss, val_wer, val_cer, val_bleu, train_losses, val_losses, epochs_list)
        raise
    finally:
        # Always clean up resources
        print("Training complete or interrupted, cleaning up...")
        cleanup()

# Script entry point
if __name__ == '__main__':
    # Register signal handler and cleanup function
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)
    
    # Run main function
    main()