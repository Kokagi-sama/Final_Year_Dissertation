# dataset_loader/lip_dataset.py
"""
Lip reading dataset processing module for video-based speech recognition.

This module provides a PyTorch dataset implementation for processing video data
of lip movements along with their corresponding text annotations. It handles loading,
preprocessing, padding, and augmentation of video frames, as well as conversion
between text and numerical representations. The dataset also includes utility methods
for calculating common speech recognition evaluation metrics (WER, CER, BLEU).
"""
from typing import List
import editdistance
from .base_dataset import BaseDataset
from .transforms import HorizontalFlip, ColorNormalize
import torch
import numpy as np
import cv2
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu

class LipDataset(BaseDataset):
    """Dataset class for lip reading video and text data.
    
    This class handles loading video frames of lip movements and corresponding text
    annotations. It supports data preprocessing, augmentation, and evaluation metrics
    calculation for lip reading models.
    
    Attributes:
        letters (list): List of characters in the vocabulary, starting with space.
        config: Configuration object containing dataset parameters.
        phase (str): Dataset phase ('train' or 'val').
        vid_padding (int): Maximum length to pad video sequences to.
        txt_padding (int): Maximum length to pad text sequences to.
        videos (list): List of video file paths.
        data (list): List of tuples containing (video_path, speaker_id, file_name).
    """
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, config, phase: str):
        super().__init__()
        # Store configuration and phase
        self.config = config
        self.phase = phase

        # Get padding lengths from config
        self.vid_padding = config.data.vid_padding
        self.txt_padding = config.data.txt_padding
        
        # Load file list based on phase (train or validation)
        file_list = config.data.train_list if phase == 'train' else config.data.val_list
        with open(file_list, 'r') as f:
            self.videos = [Path(config.data.video_path) / line.strip() for line in f.readlines()]
        
        # Process video paths to extract speaker and file information
        self.data = []
        for vid in self.videos:
            items = vid.parts
            self.data.append((vid, items[-4], items[-1])) # (video_path, speaker_id, file_name)
            
    def __len__(self):
        """Return the number of samples in the dataset.
        
        Returns:
            int: Total number of video samples.
        """
        return len(self.data)
        
    def __getitem__(self, idx):
        """Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing:
                - 'vid': Tensor of video frames [C, T, H, W]
                - 'txt': Tensor of text indices
                - 'txt_len': Original length of text before padding
                - 'vid_len': Original number of frames before padding
        """
        try:
            # Extract data for the given index
            vid_path, spk, name = self.data[idx]

            # Load video frames and annotation
            vid = self._load_video(vid_path)
            anno = self._load_annotation(
                self.config.data.anno_path / spk / 'align' / f'{name}.align'
            )

            # Apply data augmentation for training phase
            if self.phase == 'train':
                vid = HorizontalFlip(vid) # Randomly flip video horizontally
            
            vid = ColorNormalize(vid) # Normalize color values to [0,1]
            
            # Store original lengths before padding
            vid_len = vid.shape[0]
            anno_len = anno.shape[0]
            
            # Pad sequences to fixed length
            vid = self._padding(vid, self.vid_padding)
            anno = self._padding(anno, self.txt_padding)
            
            return {
                'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), # [C, T, H, W] format
                'txt': torch.LongTensor(anno),
                'txt_len': anno_len,
                'vid_len': vid_len
            }
        finally:
            self._cleanup() # Ensure cleanup happens even if an error occurs
            
    def _load_video(self, path: Path) -> np.ndarray:
        """Load video frames from directory of image files.
        
        Args:
            path (Path): Path to directory containing frame images.
            
        Returns:
            np.ndarray: Stacked frames as a 4D array with shape (T, H, W, C).
        """
        # Get all jpg files and sort them numerically by frame number
        files = sorted(
            [f for f in path.glob('*.jpg')],
            key=lambda x: int(x.stem)
        )
        
        # Load and resize each frame
        frames = []
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                img = cv2.resize(img, (128, 64), interpolation=cv2.INTER_LANCZOS4)
                frames.append(img)
        
        # Stack frames along first dimension and convert to float32
        return np.stack(frames, axis=0).astype(np.float32)
        
    def _load_annotation(self, path: Path) -> np.ndarray:
        """Load and process text annotation file.
        
        Args:
            path (Path): Path to alignment annotation file.
            
        Returns:
            np.ndarray: Array of character indices.
        """
        with open(path, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]

            # Extract words from alignment file (third column)
            txt = [line[2] for line in lines]

            # Filter out silence and pause markers
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))

        # Convert text to array of indices, starting from index 1  
        return self.txt2arr(' '.join(txt).upper(), 1)
        
    def _padding(self, array: np.ndarray, length: int) -> np.ndarray:
        """Pad array to specified length with zeros.
        
        Args:
            array (np.ndarray): Array to pad.
            length (int): Target length after padding.
            
        Returns:
            np.ndarray: Zero-padded array.
        """
        # Convert to list of arrays for easier handling
        array = [array[i] for i in range(array.shape[0])]
        size = array[0].shape

        # Add zero arrays until reaching target length
        for _ in range(length - len(array)):
            array.append(np.zeros(size))
            
        # Stack back into a single array
        return np.stack(array, axis=0)
    
    @classmethod
    def txt2arr(cls, txt: str, start: int) -> np.ndarray:
        """Convert text string to array of character indices.
        
        Args:
            txt (str): Text string to convert.
            start (int): Starting index offset (typically 1 to reserve 0 for padding).
            
        Returns:
            np.ndarray: Array of character indices.
        """
        return np.array([cls.letters.index(c) + start for c in list(txt)])
        
    @classmethod
    def arr2txt(cls, arr: np.ndarray, start: int) -> str:
        """Convert array of indices back to text string.
        
        Args:
            arr (np.ndarray): Array of character indices.
            start (int): Starting index offset used in txt2arr.
            
        Returns:
            str: Reconstructed text string.
        """
        return ''.join([cls.letters[n - start] for n in arr if n >= start]).strip()
    
    @classmethod
    def ctc_arr2txt(cls, arr: np.ndarray, start: int) -> str:
        """Convert CTC output array to text by removing duplicates.
        
        Handles CTC's blank and repeated character removal convention.
        
        Args:
            arr (np.ndarray): CTC output array of character indices.
            start (int): Starting index offset used in txt2arr.
            
        Returns:
            str: Reconstructed text with duplicates removed.
        """
        pre = -1 # Previous character index
        txt = []
        for n in arr:
            # Only add character if it's different from previous and not a blank
            if pre != n and n >= start:
                # Skip double spaces
                if len(txt) > 0 and txt[-1] == ' ' and cls.letters[n - start] == ' ':
                    continue
                txt.append(cls.letters[n - start])
            pre = n
        return ''.join(txt).strip()
            
    @staticmethod
    def wer(predict: List[str], truth: List[str]) -> List[float]:
        """Calculate Word Error Rate for each prediction-truth pair.
        
        WER = (number of word substitutions, insertions, deletions) / (number of words in truth)
        
        Args:
            predict (List[str]): List of predicted text strings.
            truth (List[str]): List of ground truth text strings.
            
        Returns:
            List[float]: List of WER values, one per sample.
        """
        word_pairs = [(p.split(' '), t.split(' ')) for p, t in zip(predict, truth)]
        return [1.0 * editdistance.eval(p, t) / len(t) for p, t in word_pairs]
        
    @staticmethod
    def cer(predict: List[str], truth: List[str]) -> List[float]:
        """Calculate Character Error Rate for each prediction-truth pair.
        
        CER = (number of character substitutions, insertions, deletions) / (number of characters in truth)
        
        Args:
            predict (List[str]): List of predicted text strings.
            truth (List[str]): List of ground truth text strings.
            
        Returns:
            List[float]: List of CER values, one per sample.
        """
        return [1.0 * editdistance.eval(p, t) / len(t) for p, t in zip(predict, truth)]
    
    @staticmethod
    def bleu(predict: List[str], truth: List[str]) -> List[float]:
        """Calculate BLEU score for each prediction-truth pair.
        
        Uses 1-gram BLEU score (weights=(1.0, 0.0)) for evaluation.
        
        Args:
            predict (List[str]): List of predicted text strings.
            truth (List[str]): List of ground truth text strings.
            
        Returns:
            List[float]: List of BLEU scores, one per sample.
        """
        return [sentence_bleu([t.split()], p.split(), weights=(1.0, 0.0)) for p, t in zip(predict, truth)]
