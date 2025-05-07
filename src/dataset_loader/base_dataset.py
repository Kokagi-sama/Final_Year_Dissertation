# datasets/base_dataset.py
"""
Abstract base dataset class for standardizing dataset implementations.

This module provides a base class that all dataset implementations should inherit from,
ensuring a consistent interface across different dataset types. It combines PyTorch's
Dataset functionality with Python's abstract base class to enforce implementation of
required methods while providing common utilities like memory cleanup.
"""
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
import gc

class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets.
    
    This class should be inherited by all dataset implementations to ensure
    consistent interface and functionality. It requires implementations to
    define __len__ and __getitem__ methods, and provides memory cleanup
    utilities.
    
    Inherits from:
        torch.utils.data.Dataset: PyTorch's dataset interface
        ABC: Python's Abstract Base Class
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __len__(self):
        """Return the total number of samples in the dataset.
        
        Must be implemented by child classes.
        
        Returns:
            int: Number of samples in the dataset
        """
        pass
        
    @abstractmethod
    def __getitem__(self, idx):
        """Retrieve a specific sample from the dataset.
        
        Must be implemented by child classes.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Any: The sample data at the given index
        """
        pass
        
    def _cleanup(self):
        """Clean up any memory that might be held.
        
        Performs garbage collection and clears CUDA cache to prevent memory leaks.
        Should be called when dataset operations are complete.
        """
        gc.collect()  # Force Python garbage collection
        torch.cuda.empty_cache()  # Clear CUDA cache if using GPU