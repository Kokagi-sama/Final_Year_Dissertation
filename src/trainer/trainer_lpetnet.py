# trainer/trainer_lpetnet.py
"""
Trainer module for LPETNet lip reading model.

This module implements the training and validation pipeline for the LPETNet architecture,
providing functionality for model training, validation, metric calculation, and checkpointing.
It includes support for different CTC decoding strategies and visualization of training progress.
"""
import torch
from torchaudio.models.decoder import ctc_decoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import numpy as np
from typing import Optional, Tuple, Dict, List
import gc
from dataset_loader.lip_dataset import LipDataset
import matplotlib.pyplot as plt
import os

class LPETNetTrainer:
    """Trainer class for the LPETNet lip reading model.
    
    This class handles the training loop, validation, metrics calculation,
    and model checkpointing for the LPETNet architecture. It supports multiple
    decoding strategies including greedy decoding and beam search with language models.
    
    Attributes:
        model: Base model for parameter updates
        net: Parallelized model for forward passes
        config: Configuration dictionary containing training parameters
        device: Device to use for training (GPU or CPU)
        wer: Current Word Error Rate
        cer: Current Character Error Rate
        ctr_tot: Total training iterations counter
        criterion: Loss function (CTC)
        optimizer: Optimization algorithm
    """
    def __init__(
        self, 
        model: nn.Module, 
        net: nn.Module, 
        config: Dict, 
        wer: float = None, 
        cer: float = None,
        optimizer_state_dict: Dict = None,
        ctr_tot: int = None
    ):
        # Store model references
        self.model = model  # Base model for parameter updates
        self.net = net  # Parallelized model for forward passes
        self.config = config

        # Set up device for training
        self.device = torch.device(f"cuda:{config.training.gpu}" if torch.cuda.is_available() else "cpu")

        # Initialize metrics with defaults or provided values
        self.wer = float('inf') if wer is None else wer
        self.cer = float('inf') if cer is None else cer
        self.ctr_tot = 0 if ctr_tot is None else ctr_tot
        
        # Move models to the appropriate device
        self.model = self.model.to(self.device)
        self.net = self.net.to(self.device)
        
        # Initialize CTC loss function
        self.criterion = nn.CTCLoss().to(self.device)

        # Get the required base_lr
        base_lr = config.training.peak_lr

        # Initialize Adam optimizer with fused implementation for faster training
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=config.training.weight_decay,
            fused=True
        )
        
        # Load optimizer state if provided (for resuming training)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
            print(f"Optimizer state loaded")
            
        # Print counter status for resumed training
        if self.ctr_tot > 0:
            print(f"Training will continue from iteration {self.ctr_tot}")
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            float: Average loss value for the epoch
        """
        total_loss = 0
        train_wer = []
        train_cer = []
        start_time = time.time()
        
        for i_iter, batch in enumerate(train_loader):
            try:
                # Set model to train mode
                self.model.train()
                self.net.train()
                
                # Move data to device (with non_blocking for async transfer)
                videos = batch['vid'].to(self.device, non_blocking=True)
                texts = batch['txt'].to(self.device, non_blocking=True)
                vid_lengths = batch['vid_len'].to(self.device, non_blocking=True)
                txt_lengths = batch['txt_len'].to(self.device, non_blocking=True)
                
                # Increment total iteration counter
                self.ctr_tot += 1
                
                # Forward pass using the parallelized model
                self.optimizer.zero_grad()
                outputs = self.net(videos)
                
                # Calculate CTC loss
                loss = self.criterion(
                    outputs.transpose(0, 1).log_softmax(-1),
                    texts,
                    vid_lengths.view(-1),
                    txt_lengths.view(-1)
                )
                
                # Backward pass and optimization
                loss.backward()
                if self.config.training.is_optimize:
                    self.optimizer.step()
                
                # Calculate metrics for monitoring
                predictions = self._decode_predictions(outputs, mode=self.config.decoder.mode)
                truth = self._decode_truth(texts)
                current_wer = LipDataset.wer(predictions, truth)
                current_cer = LipDataset.cer(predictions, truth)
                train_wer.extend(current_wer)
                train_cer.extend(current_cer)

                total_loss += loss.item()
                
                # Log progress at specified intervals
                if i_iter % self.config.training.display_interval == 0:
                    self._log_progress(
                        'Training',
                        epoch, i_iter, loss.item(),
                        train_wer,
                        train_cer,
                        None,
                        predictions[:5], truth[:5],
                        train_loader, start_time
                    )
                    
            finally:
                # Clean up GPU memory
                gc.collect()

        # Return average loss for the epoch   
        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader, epoch: int, repeat_factor: int = 1) -> Tuple[float, float, float, float]:
        """Validate the model on the validation set, repeating the process multiple times.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            repeat_factor: Number of times to repeat validation (default: 1)
            
        Returns:
            Tuple of (avg_loss, avg_wer, avg_cer, avg_bleu)
        """
        # Set models to evaluation mode
        self.model.eval()
        self.net.eval()

        # Initialize metrics
        total_loss = 0
        all_wer = []
        all_cer = []
        all_bleu = []
        start_time = time.time()
        
        # Total iterations will be repeat_factor * len(val_loader) but for our case it is 1 so no difference
        total_iterations = repeat_factor * len(val_loader)
        
        # Disable gradient calculation during validation
        with torch.no_grad():
            for repeat in range(repeat_factor):
                for i_iter, batch in enumerate(val_loader):
                    try:
                        # Calculate the global iteration number
                        global_iter = repeat * len(val_loader) + i_iter
                        
                        # Move data to device
                        videos = batch['vid'].to(self.device, non_blocking=True)
                        texts = batch['txt'].to(self.device, non_blocking=True)
                        vid_lengths = batch['vid_len'].to(self.device, non_blocking=True)
                        txt_lengths = batch['txt_len'].to(self.device, non_blocking=True)
                        
                        # Forward pass using the parallelized model
                        outputs = self.net(videos)  # Use net instead of model for parallelization
                        
                        # Calculate CTC loss
                        loss = self.criterion(
                            outputs.transpose(0, 1).log_softmax(-1),
                            texts,
                            vid_lengths.view(-1),
                            txt_lengths.view(-1)
                        )
                        
                        total_loss += loss.item()
                        
                        # Calculate metrics
                        predictions = self._decode_predictions(outputs, mode = self.config.decoder.mode)
                        truth = self._decode_truth(texts)
                        all_wer.extend(LipDataset.wer(predictions, truth))
                        all_cer.extend(LipDataset.cer(predictions, truth))
                        all_bleu.extend(LipDataset.bleu(predictions, truth))

                        # Log validation progress at specified intervals
                        if global_iter % self.config.validation.display_interval == 0:
                            self._log_progress(
                                'Validation',
                                epoch, global_iter, loss.item(),
                                all_wer,
                                all_cer,
                                all_bleu,
                                predictions[:5], truth[:5],
                                val_loader, start_time,
                                total_iters=total_iterations  # Pass total iterations for ETA calculation
                            )
                            
                    finally:
                        # Clean up GPU memory
                        gc.collect()
        
        # Calculate averages over all iterations
        avg_loss = total_loss / (len(val_loader) * repeat_factor)
        avg_wer = np.mean(all_wer)
        avg_cer = np.mean(all_cer)
        avg_bleu = np.mean(all_bleu)
        
        return avg_loss, avg_wer, avg_cer, avg_bleu
    
    def _decode_predictions(self, outputs: torch.Tensor, mode: str = "greedy") -> List[str]:
        """
        Decode model outputs using different decoding strategies.
        
        Args:
            outputs: Model output tensor of shape [batch_size, sequence_length, num_classes]
            mode: Decoding mode - "greedy", "char_lm", or "word_lm"
            
        Returns:
            List of decoded text predictions
        """
        # Simple greedy decoding
        if mode == "greedy":
            predictions = outputs.argmax(-1)  # Keep on GPU for argmax operation
            predictions_cpu = predictions.cpu().numpy()
            return [LipDataset.ctc_arr2txt(pred, start=1) for pred in predictions_cpu]
        
        # Character or word level language model decoding
        elif mode in ["char_lm", "word_lm"]:
            # Initialize appropriate decoder if not already done
            decoder_attr = f"decoder_{mode}"
            if not hasattr(self, decoder_attr):
                
                # Include the blank token in your tokens list
                tokens = [self.config.decoder.blank_token] + LipDataset.letters
                
                # Set up lexicon for word-level LM
                lexicon = None
                if mode == "word_lm":
                    lexicon = self.config.decoder.lexicon_file_path
                
                # Choose the appropriate LM file
                lm_file = self.config.decoder.char_lm_file_path if mode == "char_lm" else self.config.decoder.word_lm_file_path
                
                # Initialize decoder with appropriate parameters
                decoder = ctc_decoder(
                    lexicon=lexicon,
                    tokens=tokens,
                    lm=lm_file,
                    beam_size=self.config.decoder.beam_width,
                    lm_weight=self.config.decoder.alpha,
                    word_score=self.config.decoder.beta,
                    blank_token=self.config.decoder.blank_token,
                    sil_token=self.config.decoder.blank_token,
                )
                
                # Store the decoder and tokens
                setattr(self, decoder_attr, decoder)
                setattr(self, f"tokens_{mode}", tokens)
            
            # Get the appropriate decoder
            decoder = getattr(self, decoder_attr)
            tokens = getattr(self, f"tokens_{mode}")
            
            # Get log probabilities and move to CPU
            log_probs = outputs.log_softmax(-1).cpu()
            
            # Get input lengths
            batch_size = log_probs.size(0)
            input_lengths = torch.full((batch_size,), log_probs.size(1), dtype=torch.int32)
            
            try:
                # Perform beam search decoding
                hypotheses = decoder(log_probs, input_lengths)
                
                # Process results
                decoded_texts = []
                for hypothesis in hypotheses:
                    if len(hypothesis) > 0:
                        # Get best hypothesis
                        best_hyp = hypothesis[0]
                        
                        if mode == "word_lm":
                            # For word-level LM, use the words attribute
                            text = ' '.join(best_hyp.words) if best_hyp.words else ""
                        else:
                            # For character-level LM, convert tokens to characters
                            text = ''.join([tokens[idx] for idx in best_hyp.tokens if idx > 0])
                        
                        decoded_texts.append(text)
                    else:
                        # Fall back to greedy decoding if no hypothesis found
                        decoded_texts.append("")
            except Exception as e:
                print(f"Error in {mode} decoding: {e}")
                # Fall back to greedy decoding
                return self._decode_predictions(outputs, mode="greedy")
            
            return decoded_texts
        
        else:
            raise ValueError(f"Unknown decoding mode: {mode}. Use 'greedy', 'char_lm', or 'word_lm'.")
        
    def _decode_truth(self, texts: torch.Tensor) -> List[str]:
        """Decode ground truth texts.
        
        Args:
            texts: Tensor containing ground truth label indices
            
        Returns:
            List of decoded text strings
        """
        # Only move to CPU for final string processing which can't be done on GPU
        texts_cpu = texts.cpu().numpy()
        return [LipDataset.arr2txt(text, start=1) for text in texts_cpu]
        
    def save_checkpoint(self, loss: float, wer: float, cer: float, bleu: float, train_losses: list, val_losses: list, epochs_list: list):
        """Save model checkpoint with timestamp and metrics.
        
        Args:
            loss: Current validation loss
            wer: Current Word Error Rate
            cer: Current Character Error Rate
            bleu: Current BLEU score
            train_losses: History of training losses
            val_losses: History of validation losses
            epochs_list: List of completed epochs
        """
        # Create save directory if it doesn't exist
        save_path = Path(self.config.training.save_prefix)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to filename for unique identification
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f'LPETNet_{timestamp}_loss_{loss:.4f}_wer_{wer:.4f}_cer_{cer:.4f}_bleu_{bleu:.4f}.pt'
        save_path = save_path / checkpoint_name
        
        # Save model and training state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'wer': wer,
            'cer': cer,
            'bleu': bleu,
            'ctr_tot': self.ctr_tot,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_list': epochs_list,
            'timestamp': timestamp  # Also save timestamp inside the checkpoint
        }, save_path)
      
    def _log_progress(
        self,
        type: str = None,
        epoch: int = None,
        iteration: int = None,
        loss: float = None,
        wer: List[float] = None,
        cer: List[float] = None,
        bleu: Optional[List[float]] = None,
        predictions: List[str] = None,
        truth: List[str] = None,
        loader: DataLoader = None,
        start_time: float = None,
        total_iters: Optional[int] = None
    ):
        """Log training/validation progress to console.
        
        Args:
            type: Type of logging ('Training' or 'Validation')
            epoch: Current epoch number
            iteration: Current iteration number
            loss: Current loss value
            wer: List of Word Error Rates
            cer: List of Character Error Rates
            bleu: List of BLEU scores (optional)
            predictions: List of model predictions to display
            truth: List of ground truth texts to display
            loader: DataLoader being used
            start_time: Time when the epoch/validation started
            total_iters: Total number of iterations for ETA calculation
        """
        # Calculate ETA (estimated time of arrival/completion)
        elapsed_time = time.time() - start_time
        time_per_iter = elapsed_time / (iteration + 1)
        remaining_iters = (total_iters if total_iters is not None else len(loader)) - iteration - 1
        eta = time_per_iter * remaining_iters / 3600.0 # Convert to hours
        
        # Print comparison table header
        print('=' * 101)
        print(f'{"Prediction":<50}|{"Truth":>50}')
        print('=' * 101)

        # Print prediction vs truth comparison
        for pred, true in zip(predictions[:5], truth[:5]):
            print(f'{pred:<50}|{true:>50}')
        print('=' * 101)
        
        # Print metrics with or without BLEU score
        if bleu is not None:
            print(f'Epoch: {epoch + 1}, {type}_Iteration: {iteration}/{total_iters if total_iters is not None else len(loader)}, ETA: {eta:.2f}h, Loss: {loss:.4f}, WER: {np.mean(wer):.4f}, CER: {np.mean(cer):.4f}, BLEU: {np.mean(bleu):.4f}')
        else:
            print(f'Epoch: {epoch + 1}, {type}_Iteration: {iteration}/{total_iters if total_iters is not None else len(loader)}, ETA: {eta:.2f}h, Loss: {loss:.4f}, WER: {np.mean(wer):.4f}, CER: {np.mean(cer):.4f}')
        print('=' * 101)

    def plot_losses(self, train_losses, val_losses, epochs):
        """Plot and save training and validation loss curves.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            epochs: List of epoch numbers
        """
        # Create figure with appropriate size
        plt.figure(figsize=(10, 6))

        # Plot training losses as blue line
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')

        # Plot validation losses as red dots
        plt.plot(epochs, val_losses, 'r.', label='Validation Loss')

        # Add title and labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Create directory if it doesn't exist
        save_dir = self.config.training.loss_path
        os.makedirs(save_dir, exist_ok=True)
        
        # Save figure with epoch number in filename
        plt.savefig(save_dir / f'loss_plot_epoch_{epochs[-1]}.png')
        plt.close()