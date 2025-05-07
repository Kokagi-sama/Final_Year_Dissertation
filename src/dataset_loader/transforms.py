# dataset_loader/transforms.py
"""
Image transformation functions for data augmentation in lip reading models.

This module provides various transformation functions for augmenting image data
during training. Data augmentation increases the diversity of training examples
without collecting additional data, helping improve model generalization and
robustness against variations in real-world inputs.

The module currently implements:
- Horizontal flipping: Mirrors images along the vertical axis
- Color normalization: Scales pixel values to the range [0,1]

Each transformation function takes a batch of images as input and returns
the transformed batch, making them suitable for use in preprocessing pipelines.
"""
import random
import numpy as np

def HorizontalFlip(batch_img: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Horizontally flips the images with probability p.
    
    Args:
        batch_img (np.ndarray): Batch of images of shape [batch, height, width, channels]
        p (float, optional): Probability of applying the flip. Defaults to 0.5.
        
    Returns:
        np.ndarray: Batch of images, potentially flipped horizontally
    """
    if random.random() > p:
        return batch_img[:,:,::-1,...]
    return batch_img

def ColorNormalize(batch_img: np.ndarray) -> np.ndarray:
    """Normalizes color values to range [0,1].
    
    Args:
        batch_img (np.ndarray): Batch of images of shape [batch, height, width, channels]
        
    Returns:
        np.ndarray: Batch of normalized images with values in range [0,1]
    """
    return batch_img / 255.0