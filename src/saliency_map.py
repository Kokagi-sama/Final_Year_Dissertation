# saliency_map.py
"""
Saliency map visualization script for the ALPETNet lip reading model. 
(Can be used for LPETNet too though some configurations' changes must be made).

This script loads a trained ALPETNet model, processes video frames,
and generates saliency maps to visualize which regions of the input
frames most significantly influence the model's predictions. The
visualization helps interpret how the model focuses on specific
facial features during lip reading.
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from models.ALPETNet import ALPETNet
from trainer.trainer_comparative_analysis import ALPETNetTrainer
from config.experiment_config import get_config

def load_frames_from_directory(directory_path):
    """Load video frames from a directory of image files.
    
    Args:
        directory_path (str or Path): Path to directory containing frame images.
            Images should be named in a way that allows numerical sorting.
            
    Returns:
        numpy.ndarray: Stacked frames as a 4D array with shape (num_frames, height, width, channels),
            with values converted to float32 type.
    """
    directory = Path(directory_path)

    # Sort files numerically by frame number in filename
    frame_files = sorted(directory.glob('*.jpg'), key=lambda x: int(x.stem))
    
    frames = []
    for frame_file in frame_files:
        # Load the image using OpenCV
        img = cv2.imread(str(frame_file))
        if img is not None:
            # Resize to standard dimensions expected by the model
            img = cv2.resize(img, (128, 64), interpolation=cv2.INTER_LANCZOS4)
            frames.append(img)
    
    # Stack frames along the first dimension
    return np.stack(frames, axis=0).astype(np.float32)

if __name__ == "__main__":
    # Get configuration from config module
    config = get_config()
    
    # Initialize model with dropout settings from config
    model = ALPETNet(
        dropout_p=config.model.cnn_dropout,
        transformer_dropout=config.model.trans_dropout
    )
    
    # Load pretrained weights if specified in config
    if config.model.pretrained_path:
        checkpoint_path = config.model.pretrained_path

        # Load model weights from checkpoint file
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Set up device (GPU or CPU) for model inference
    device = torch.device(f"cuda:{config.training.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create parallel model version for multi-GPU support
    net = torch.nn.DataParallel(model).to(device)
    
    # Initialize trainer that contains saliency map visualization functionality
    trainer = ALPETNetTrainer(model=model, net=net, config=config)
    
    # Get directory containing input frames from config
    frames_directory = config.analysis.frames_directory

    # Load video frames from the specified directory
    frames = load_frames_from_directory(frames_directory)
    
    # Generate saliency map and get predicted text
    predicted_text = trainer.visualize_saliency_map_from_frames(
        frames=frames,
    )
    
    # Print the model's predicted text for the input frames
    print(f"Predicted text: {predicted_text}")
