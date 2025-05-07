# trainer/trainer_comparative-analysis.py
"""
Comparative analysis trainer module for ALPETNet lip reading model 
(Can be used for LPETNet too though some configurations' changes must be made).

This module implements specialized training and analysis features for the ALPETNet 
architecture, including confusion matrix generation and saliency map visualization. 
It extends standard training functionality with tools for detailed model inspection 
and performance analysis across different phoneme and character recognition patterns.
"""
import cv2
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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import nltk
from nltk.corpus import cmudict
from skimage.color import rgb2gray
import os
import matplotlib.cm as cm

class ALPETNetTrainer:
    """Trainer class for the ALPETNet lip reading model with analysis capabilities.
    
    This class handles the training, validation, and analysis of the ALPETNet model,
    with specialized features for generating confusion matrices and saliency maps.
    It supports multiple decoding strategies and provides detailed visualization
    tools for model interpretation.
    
    Attributes:
        model: Base model for parameter updates
        net: Parallelized model for forward passes
        config: Configuration dictionary with training parameters
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
           
    def validate(self, val_loader: DataLoader, epoch: str, repeat_factor: int = 1) -> Tuple[float, float, float, float, np.ndarray]:
        """Validate the model on the validation set, repeating the process multiple times.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            repeat_factor: Number of times to repeat validation (default: 1)
            
        Returns:
            Tuple of (avg_loss, avg_wer, avg_cer, avg_bleu, confusion_matrix)
        """
        # Set models to evaluation mode
        self.model.eval()
        self.net.eval()

        # Initialize metrics
        total_loss = 0
        all_wer = []
        all_cer = []
        all_bleu = []
        all_predictions = []
        all_truths = []
        
        # Disable gradient calculation during validation
        with torch.no_grad():
            # Repeat the validation process repeat_factor times for more stable metrics
            for repeat in range(repeat_factor):
                for i_iter, batch in enumerate(val_loader):
                    try:                       
                        # Move data to device
                        videos = batch['vid'].to(self.device, non_blocking=True)
                        texts = batch['txt'].to(self.device, non_blocking=True)
                        vid_lengths = batch['vid_len'].to(self.device, non_blocking=True)
                        txt_lengths = batch['txt_len'].to(self.device, non_blocking=True)
                        
                        # Forward pass using the parallelized model
                        outputs = self.net(videos)

                        # Calculate CTC loss
                        loss = self.criterion(
                            outputs.transpose(0, 1).log_softmax(-1),
                            texts,
                            vid_lengths.view(-1),
                            txt_lengths.view(-1)
                        )
                        
                        total_loss += loss.item()
                        
                        # Calculate metrics
                        predictions = self._decode_predictions(outputs, mode=self.config.decoder.mode)
                        truth = self._decode_truth(texts)
                        all_predictions.extend(predictions)
                        all_truths.extend(truth)
                        
                        all_wer.extend(LipDataset.wer(predictions, truth))
                        all_cer.extend(LipDataset.cer(predictions, truth))
                        all_bleu.extend(LipDataset.bleu(predictions, truth))
                        
                           
                    finally:
                        # Clean up GPU memory
                        gc.collect()
        
        # Generate character confusion matrix data
        char_confusion_matrix = self._generate_confusion_matrix_data(all_predictions, all_truths)
        
        # Add predictions and truth to the returned data for phoneme matrix generation
        char_confusion_matrix['predictions'] = all_predictions
        char_confusion_matrix['truth'] = all_truths
        
        # Calculate averages over all iterations
        avg_loss = total_loss / (len(val_loader) * repeat_factor)
        avg_wer = np.mean(all_wer)
        avg_cer = np.mean(all_cer)
        avg_bleu = np.mean(all_bleu)
        
        return avg_loss, avg_wer, avg_cer, avg_bleu, char_confusion_matrix

    
    def _decode_predictions(self, outputs: torch.Tensor, mode: str = "greedy") -> List[str]:
        """Decode model outputs using different decoding strategies.
        
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
    
    def _generate_confusion_matrix_data(self, predictions: List[str], truth: List[str]) -> np.ndarray:
        """Generate character-wise confusion matrix data focusing on alphabetic characters.
        
        Args:
            predictions: List of predicted text strings
            truth: List of ground truth text strings
            
        Returns:
            Dictionary containing normalized confusion matrix and labels
        """
        
        # Define alphabet characters a-z plus space
        alphabet = list('abcdefghijklmnopqrstuvwxyz ')
        char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        
        # Collect all character pairs for confusion matrix
        true_chars = []
        pred_chars = []
        
        for true_text, pred_text in zip(truth, predictions):
            # Convert to lowercase to focus on a-z
            true_text = true_text.lower()
            pred_text = pred_text.lower()
            
            # Align characters between prediction and truth
            for i in range(min(len(true_text), len(pred_text))):
                if true_text[i] in alphabet:
                    true_chars.append(true_text[i])
                    # If predicted char is in alphabet, use it; otherwise, use a placeholder
                    if pred_text[i] in alphabet:
                        pred_chars.append(pred_text[i])
                    else:
                        pred_chars.append('?')  # Use ? for non-alphabet predictions
            
            # Handle remaining characters in truth
            for i in range(len(pred_text), len(true_text)):
                if true_text[i] in alphabet:
                    true_chars.append(true_text[i])
                    pred_chars.append('?')  # Missing in prediction
            
            # Handle remaining characters in prediction
            for i in range(len(true_text), len(pred_text)):
                if pred_text[i] in alphabet:
                    pred_chars.append(pred_text[i])
                    true_chars.append('!')  # Missing in truth
        
        # Add placeholder characters to alphabet
        for placeholder in ['?', '!']:
            if placeholder not in alphabet:
                alphabet.append(placeholder)
                char_to_idx[placeholder] = len(alphabet) - 1

        # Convert characters to indices
        true_indices = [char_to_idx[char] for char in true_chars]  # No default needed
        pred_indices = [char_to_idx[char] for char in pred_chars]  # No default needed
        
        # Create confusion matrix
        cm = confusion_matrix(true_indices, pred_indices, labels=range(len(alphabet)))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        # Replace space with visible label for display
        display_labels = [char if char != ' ' else '␣' for char in alphabet]
        
        return {
            'matrix': cm_normalized,
            'labels': display_labels
        }

    def _generate_phoneme_confusion_matrix(self, predictions: List[str], truth: List[str]) -> dict:
        """Generate phoneme-wise confusion matrix using ARPABET phonemes.
        
        Args:
            predictions: List of predicted text strings
            truth: List of ground truth text strings
            
        Returns:
            Dictionary containing normalized confusion matrix and labels
        """
        
        # Download CMUDict if not already present
        try:
            cmu_dict = cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            cmu_dict = cmudict.dict()
        
        # Use only the specified phonemes
        specified_phonemes = [
            'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'eh', 'ey', 'f', 'g', 
            'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ow', 'p', 'r', 's', 't', 'th', 
            'uw', 'v', 'w', 'y', 'z'
        ]
        
        phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(specified_phonemes)}
        
        # Function to convert word to phonemes
        def word_to_phonemes(word):
            word = word.lower().strip()
            if not word:
                return []
            
            # Handle words with multiple pronunciations by taking the first one
            if word in cmu_dict:
                # Get first pronunciation and remove stress markers
                raw_phonemes = cmu_dict[word][0]
                # Extract base phonemes without stress markers and filter to only include specified phonemes
                filtered_phonemes = []
                for p in raw_phonemes:
                    # Remove stress markers (numbers)
                    base_phoneme = ''.join([c for c in p.lower() if not c.isdigit()])
                    if base_phoneme in specified_phonemes:
                        filtered_phonemes.append(base_phoneme)
                return filtered_phonemes
            
            # For words not in dictionary, return empty list
            return []
        
        # Collect phoneme pairs for confusion matrix
        true_phonemes = []
        pred_phonemes = []
        
        for true_text, pred_text in zip(truth, predictions):
            # Split into words
            true_words = true_text.lower().split()
            pred_words = pred_text.lower().split()
            
            # Convert words to phonemes
            true_word_phonemes = [word_to_phonemes(word) for word in true_words]
            pred_word_phonemes = [word_to_phonemes(word) for word in pred_words]
            
            # Flatten word phonemes into single lists
            true_all_phonemes = [p for word_phons in true_word_phonemes for p in word_phons]
            pred_all_phonemes = [p for word_phons in pred_word_phonemes for p in word_phons]
            
            # Align phonemes (simplified approach)
            for i in range(min(len(true_all_phonemes), len(pred_all_phonemes))):
                if true_all_phonemes[i] in specified_phonemes and pred_all_phonemes[i] in specified_phonemes:
                    true_phonemes.append(true_all_phonemes[i])
                    pred_phonemes.append(pred_all_phonemes[i])
        
        # Convert phonemes to indices
        true_indices = [phoneme_to_idx[p] for p in true_phonemes]
        pred_indices = [phoneme_to_idx[p] for p in pred_phonemes]
        
        # Create confusion matrix
        cm = confusion_matrix(true_indices, pred_indices, labels=range(len(specified_phonemes)))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        return {
            'matrix': cm_normalized,
            'labels': specified_phonemes
        }
 
    def save_checkpoint(self, confusion_matrix_data: dict = None):
        """Save model checkpoint with timestamp and confusion matrix.
        
        Args:
            confusion_matrix_data: Dictionary containing confusion matrix data and labels
        """     
        # Add timestamp to filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
                       
        # If confusion matrix data is provided, save it separately
        if confusion_matrix_data is not None:
            # Save character confusion matrix
            self._save_confusion_matrix(confusion_matrix_data, timestamp)
            
            # Generate and save phoneme confusion matrix
            phoneme_confusion_data = self._generate_phoneme_confusion_matrix(
                confusion_matrix_data['predictions'], 
                confusion_matrix_data['truth']
            )
            self._save_phoneme_confusion_matrix(
                phoneme_confusion_data, 
                timestamp, 
            )

    def _save_confusion_matrix(self, confusion_matrix_data: dict, timestamp: str):
        """Save the confusion matrix as a heatmap image with improved visualization.
        
        Args:
            confusion_matrix_data: Dictionary containing matrix data and labels
            timestamp: Timestamp string for the filename
        """
        
        matrix = confusion_matrix_data['matrix']
        labels = confusion_matrix_data['labels']
        
        save_dir = self.config.analysis.confusion_matrix_output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate figure size based on number of labels - ensure square aspect ratio
        n_labels = len(labels)
        fig_size = max(12, n_labels * 0.5)  # Scale figure size with number of labels
        
        # Plot confusion matrix - use same value for width and height to ensure square
        plt.figure(figsize=(fig_size, fig_size))
        
        # Adjust font size based on number of labels
        annot_size = max(5, 12 - (n_labels // 10))  # Decrease font size as labels increase
        
        # Use annot=False for very large matrices to avoid cluttering
        use_annot = n_labels <= 40
        
        # Create heatmap with adjusted parameters
        ax = sns.heatmap(
            matrix, 
            annot=False, 
            fmt='.2f' if use_annot else '', 
            cmap='Blues',
            xticklabels=labels, 
            yticklabels=labels,
            annot_kws={'size': annot_size} if use_annot else {},
            linewidths=0.01 if n_labels > 30 else 0.5,  # Thinner lines for large matrices
            cbar_kws={'shrink': 0.7},  # Smaller colorbar
            square=True  # Force square cells in the heatmap
        )
        
        # Increase tick label size - set a minimum size to ensure readability
        # Adjust this value based on your preference
        tick_size = max(10, 14 - (n_labels // 30))  # Larger minimum size, slower decrease rate
        
        plt.xticks(rotation=90, fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        
        plt.xlabel('Predicted', fontsize=14)  # Increased from 12
        plt.ylabel('True', fontsize=14)  # Increased from 12
        plt.title(f'Character Confusion Matrix', fontsize=16)  # Increased from 14
        
        # Tight layout to use space efficiently
        plt.tight_layout()
        
        # Save the figure with high DPI for clarity
        plt.savefig(save_dir / f'alphabet_confusion_matrix_{timestamp}.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # Also save the raw confusion matrix data
        np.save(save_dir / f'alphabet_confusion_matrix_{timestamp}.npy', matrix)
        
    def _save_phoneme_confusion_matrix(self, confusion_matrix_data: dict, timestamp: str):
        """Save the phoneme confusion matrix as a heatmap image.
        
        Args:
            confusion_matrix_data: Dictionary containing matrix data and labels
            timestamp: Timestamp string for the filename
        """
        
        matrix = confusion_matrix_data['matrix']
        labels = confusion_matrix_data['labels']
        
        # Create directory if it doesn't exist
        save_dir = self.config.analysis.confusion_matrix_output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate figure size based on number of labels
        n_labels = len(labels)
        fig_size = max(15, n_labels * 0.4)  # Scale figure size with number of labels
        
        # Plot confusion matrix - use same value for width and height to ensure square
        plt.figure(figsize=(fig_size, fig_size))
        
        # Adjust font size based on number of labels
        annot_size = max(4, 10 - (n_labels // 15))  # Decrease font size as labels increase
        
        # Use annot=False for very large matrices to avoid cluttering
        use_annot = n_labels <= 50
        
        # Create heatmap with adjusted parameters
        ax = sns.heatmap(
            matrix, 
            annot=False, 
            fmt='.2f' if use_annot else '', 
            cmap='Blues',
            xticklabels=labels, 
            yticklabels=labels,
            annot_kws={'size': annot_size} if use_annot else {},
            linewidths=0.01 if n_labels > 30 else 0.5,  # Thinner lines for large matrices
            cbar_kws={'shrink': 0.7},  # Smaller colorbar
            square=True  # Force square cells in the heatmap
        )
        
        # Increase tick label size - set a minimum size to ensure readability
        # Adjust this value based on your preference
        tick_size = max(10, 14 - (n_labels // 30))  # Larger minimum size, slower decrease rate
        
        plt.xticks(rotation=90, fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        
        plt.xlabel('Predicted', fontsize=14)  # Increased from 12
        plt.ylabel('True', fontsize=14)  # Increased from 12
        plt.title(f'Phoneme Confusion Matrix (ARPABET)', fontsize=16)  # Increased from 14
        
        # Tight layout to use space efficiently
        plt.tight_layout()
        
        # Save the figure with high DPI for clarity
        plt.savefig(save_dir / f'phoneme_confusion_matrix_{timestamp}.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # Also save the raw confusion matrix data
        np.save(save_dir / f'phoneme_confusion_matrix_{timestamp}.npy', matrix)

    def generate_saliency_map(self, video_frames, text=None):
        """Generate saliency maps for video frames using guided backpropagation.
        
        Args:
            video_frames: Input frames tensor
            text: Optional target text for the saliency calculation
            
        Returns:
            Tuple of (saliency_maps, predicted_text, alignment)
        """
        # Temporarily set model to train mode
        was_training = self.model.training
        self.model.train()
        
        # Add batch dimension if not present
        if len(video_frames.shape) == 4:
            video_frames = video_frames.unsqueeze(0)
        
        # Create a copy of the video that requires gradient
        video_input = video_frames.clone().detach().to(self.device)
        video_input.requires_grad = True
        
        try:
            # Forward pass
            with torch.enable_grad():
                outputs = self.model(video_input)
                
                # Get CTC alignment using greedy decoding
                predictions = outputs.argmax(-1)  # [batch_size, sequence_length]
                
                # Convert predictions to text
                predicted_text = [LipDataset.ctc_arr2txt(pred.cpu().numpy(), start=1) for pred in predictions]
                
                # Sum the probabilities of the predicted alignment
                alignment_probs = torch.zeros(1, device=self.device)
                for b in range(outputs.size(0)):
                    for t in range(outputs.size(1)):
                        alignment_probs += outputs[b, t, predictions[b, t]]
                
                # Compute gradients
                alignment_probs.backward()
            
            # Apply guided backpropagation - zero out negative gradients
            with torch.no_grad():
                guided_grads = video_input.grad.clone()
                guided_grads = torch.clamp(guided_grads, min=0)
                
                # Compute saliency as the absolute value and max across channels
                saliency = guided_grads.abs().max(dim=1)[0]  # Max across RGB channels
                
                # Normalize saliency maps for visualization
                saliency_normalized = []
                for i in range(saliency.size(0)):
                    # Normalize each frame's saliency map
                    frame_saliencies = []
                    for t in range(saliency.size(1)):
                        frame_saliency = saliency[i, t]
                        if frame_saliency.max() > 0:
                            frame_saliency = frame_saliency / frame_saliency.max()
                        frame_saliencies.append(frame_saliency)
                    saliency_normalized.append(torch.stack(frame_saliencies))
            
            return torch.stack(saliency_normalized), predicted_text, predictions
        
        finally:
            # Restore the model's original mode
            self.model.train(was_training)

    def visualize_saliency_map_from_frames(self, frames, word=None, speaker=None, threshold=0.0):
        """Generate and save saliency map visualization for a sequence of frames.
        
        Args:
            frames: List of image frames or numpy array of shape [sequence_length, height, width, channels]
            word: Optional specific word to visualize
            speaker: Optional speaker ID
            threshold: Threshold value (0-1) to filter saliency map
            
        Returns:
            str: Predicted text for the input frames
        """        
        # Convert frames to tensor if needed
        if isinstance(frames, list):
            frames = np.stack(frames, axis=0)
        
        # Ensure frames are in the right format [sequence_length, height, width, channels]
        if frames.shape[-1] != 3:
            frames = np.transpose(frames, (0, 2, 3, 1))
        
        # Normalize frames if needed (assuming 0-255 range)
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        # Convert to tensor and transpose to [channels, sequence_length, height, width]
        video_tensor = torch.FloatTensor(frames.transpose(3, 0, 1, 2))
        
        # Generate saliency maps
        saliency_maps, predicted_text, alignment = self.generate_saliency_map(video_tensor)
        
        # Convert tensors to numpy for visualization
        saliency_maps = saliency_maps[0].cpu().numpy()  # [seq_len, height, width]
        
        # Get CTC alignment with blanks for visualization
        alignment = alignment[0].cpu().numpy()
        alignment_text = []
        for idx in alignment:
            if idx == 0:  # CTC blank
                alignment_text.append('␣')
            else:
                alignment_text.append(LipDataset.letters[idx-1])
        
        # Create directory for saving visualizations
        save_dir = self.config.analysis.saliency_map_output_dir
        os.makedirs(save_dir, exist_ok=True)

        output_path = save_dir / "saliency_map_output.png"
        
        # Determine number of frames to visualize
        n_frames = min(len(frames), len(saliency_maps), 75)  # Limit to 75 frames
        
        # Create figure for visualization with 2 rows
        fig, axes = plt.subplots(2, n_frames, figsize=(n_frames*2, 6))
        
        # Get a colormap for the saliency (blue for high values)
        hot_cmap = cm.get_cmap('Blues')
        
        # Plot frames with saliency overlay on top row and original frames on bottom row
        for i in range(n_frames):
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(frames[i].astype(np.float32), cv2.COLOR_BGR2RGB)
            
            # TOP ROW: Display saliency maps overlaid on frames
            # Display the original RGB frame
            axes[0, i].imshow(rgb_frame)
            
            # Apply threshold to saliency map
            saliency = saliency_maps[i].copy()
            saliency[saliency < threshold] = 0  # Only show values above threshold

            # Convert to 8-bit for CLAHE
            saliency_8bit = (saliency * 255).astype(np.uint8)

            # Apply CLAHE to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            saliency_clahe = clahe.apply(saliency_8bit)

            # Apply Gaussian blur to smooth the saliency map
            saliency_blurred = cv2.GaussianBlur(saliency_clahe, (5, 5), sigmaX=1.0)

            # Normalize back to 0-1 range for visualization
            saliency = saliency_blurred / 255.0
            
            # Create a colored saliency map with transparency
            if saliency.max() > 0:  # Normalize only if there are non-zero values
                saliency = saliency / saliency.max()
            
            # Apply colormap to create RGBA image (with alpha channel)
            colored_saliency = hot_cmap(saliency)
            
            # Make low values transparent
            colored_saliency[..., 3] = saliency  # Alpha channel based on saliency values
            
            # Overlay saliency on the original frame
            axes[0, i].imshow(colored_saliency, alpha=1.0)
            axes[0, i].axis('off')
            axes[0, i].set_title(alignment_text[i])
            
            # BOTTOM ROW: Display original frames
            axes[1, i].imshow(rgb_frame)
            axes[1, i].axis('off')
        
        # Set overall title
        title = f"Saliency map for '{predicted_text[0]}'"
        if word:
            title += f" (word: {word})"
        if speaker:
            title += f" (speaker: {speaker})"
        fig.suptitle(title, fontsize=16)
        
        # Add row labels
        fig.text(0.01, 0.75, 'Saliency\nOverlay', ha='left', va='center', fontsize=14, fontweight='bold')
        fig.text(0.01, 0.25, 'Original\nFrames', ha='left', va='center', fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.05)  # Make space for row labels
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            word_str = f"_{word}" if word else ""
            speaker_str = f"_speaker{speaker}" if speaker else ""
            plt.savefig(save_dir / f"saliency_map{word_str}{speaker_str}_{timestamp}.png", 
                    bbox_inches='tight', dpi=300)
        
        plt.close()
        
        return predicted_text[0]
