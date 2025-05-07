# config/experiment_config.py
"""
Experiment configuration module for ALPETNet lip reading model.

This module provides a function to retrieve the default experiment configuration
with appropriate path settings. It extends the base configuration by setting
the model save path based on the training prefix configuration.
"""
from .base_config import Config
from pathlib import Path

def get_config():
    """Get the default experiment configuration with appropriate path settings.
    
    Creates a default Config object and sets derived paths to ensure consistency
    across the application. This is the main entry point for configuration in 
    the application.
    
    Returns:
        Config: A fully initialized configuration object ready for use.
    """
    # Create default configuration object
    config = Config()
    
    # Set model save path to match the training save prefix for consistency
    config.model.save_path = Path(config.training.save_prefix)
    
    return config