# config/base_config.py
"""
Configuration classes for ALPETNet lip reading model.
(Can be used for LPETNet too though some configurations' changes must be made).

This module defines the configuration structure for the ALPETNet model, including
dataset parameters, model architecture settings, training hyperparameters, and
analysis options. It uses dataclasses for type-safe and maintainable configuration.
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

@dataclass
class DataConfig:
    """Configuration for dataset and data processing.
    
    Defines paths to data files, padding lengths, and other dataset-specific settings.
    
    Attributes:
        data_type (str): Type of dataset split to use (e.g., 'overlap').
        data_root (Path): Root directory for dataset files.
        video_path (Path): Directory containing video files.
        anno_path (Path): Directory containing annotation files.
        train_list (Path): Path to file listing training samples.
        val_list (Path): Path to file listing validation samples.
        vid_padding (int): Maximum length to pad video sequences to.
        txt_padding (int): Maximum length to pad text sequences to.
    """
    data_type: str = 'overlap'
    data_root: Path = Path('./dataset/data')
    video_path: Path = Path('./dataset/Grid_videos')
    anno_path: Path = Path('./dataset/Grid_alignments')
    train_list: Path = None  # Set in __post_init__
    val_list: Path = None    # Set in __post_init__
    vid_padding: int = 75
    txt_padding: int = 200
    
    def __post_init__(self):
        """Initialize derived paths after the dataclass is instantiated."""
        # Create paths to train and validation file lists based on data_type
        self.train_list = self.data_root / f"{self.data_type}_train.txt"
        self.val_list = self.data_root / f"{self.data_type}_val.txt"

@dataclass
class ModelConfig:
    """Configuration for model architecture.
    
    Defines model hyperparameters, dropout rates, and paths for saving/loading models.
    
    Attributes:
        trans_dropout (float): Dropout rate for transformer layers.
        cnn_dropout (float): Dropout rate for CNN layers.
        d_model (int): Dimension of model features in transformer.
        nhead (int): Number of attention heads in transformer.
        random_seed (int): Seed for random number generators to ensure reproducibility.
        save_path (Optional[Path]): Directory to save model checkpoints.
        pretrained_path (Optional[Path]): Path to pretrained model for loading.
    """
    trans_dropout: float = 0.1
    cnn_dropout: float = 0.5
    d_model: int = 512
    nhead: int = 8
    random_seed: int = 0
    save_path: Optional[Path] = None
    pretrained_path: Optional[Path] = Path('./pretrain_models/ALPETNet_20250326_042518_loss_0.0608_wer_0.0416_cer_0.0171_bleu_0.9584.pt')

@dataclass
class TrainingConfig:
    """Configuration for training process.
    
    Defines batch sizes, learning rates, optimization settings, and other training parameters.
    
    Attributes:
        gpu (str): GPU device ID to use for training.
        batch_size (int): Number of samples in each training batch.
        peak_lr (float): Peak learning rate for optimizer.
        num_workers (int): Number of worker processes for data loading.
        max_epoch (int): Maximum number of training epochs.
        display_interval (int): How often to display training progress.
        save_prefix (Path): Directory prefix for saving model checkpoints.
        is_optimize (bool): Whether to apply optimizer step during training.
        weight_decay (float): L2 regularization strength.
        loss_path (Path): Path to save loss plots.
    """
    gpu: str = '0'
    batch_size: int = 64
    peak_lr: float = 1e-3
    num_workers: int = 4
    max_epoch: int = 120
    display_interval: int = 10
    save_prefix: Path = Path('./weights/')
    is_optimize: bool = True
    weight_decay: float = 0.0
    loss_path: Path = Path('./plots/ALPETNet')

@dataclass
class ValidationConfig:
    """Configuration for validation process.
    
    Defines parameters specific to validation runs.
    
    Attributes:
        display_interval (int): How often to display validation progress.
    """
    display_interval: int = 10

@dataclass
class DecoderConfig:
    """Configuration for CTC decoder.
    
    Defines parameters for different decoding strategies, including beam search settings
    and language model configurations.
    
    Attributes:
        mode (str): Decoding strategy ("greedy", "char_lm", or "word_lm").
        beam_width (int): Beam width for beam search decoder.
        alpha (float): Language model weight for beam search.
        beta (float): Word insertion bonus for beam search.
        blank_token (str): Token used for CTC blank symbol.
        char_lm_file_path (str): Path to character language model file.
        word_lm_file_path (str): Path to word language model file.
        lexicon_file_path (str): Path to lexicon file for word decoding.
    """
    mode: str = "greedy"    # Can be either greedy, char_lm or word_lm
    beam_width: int = 10
    alpha: float = 1.0
    beta: float = 1.5
    blank_token: str  = '_'
    char_lm_file_path: str = './5-gram-model/character/grid_char.binary'
    word_lm_file_path: str = './5-gram-model/word/grid_word.binary'
    lexicon_file_path: str = './5-gram-model/word/lexicon.txt'

@dataclass
class AnalysisConfig:
    """Configuration for model analysis and visualization.
    
    Defines paths for saving analysis outputs such as beam width comparative analysis,
    saliency maps, and confusion matrices.
    
    Attributes:
        decoder_output_dir (Optional[Path]): Directory to save beam width analysis results.
        saliency_map_output_dir (Optional[Path]): Directory to save saliency map visualizations.
        confusion_matrix_output_dir (Optional[Path]): Directory to save confusion matrices.
        frames_directory (str): Directory containing input frames for visualization.
    """
    decoder_output_dir: Optional[Path] = Path("./beam_width_comparative_analysis/ALPETNet")
    saliency_map_output_dir: Optional[Path] = Path("./saliency_map/ALPETNet")
    confusion_matrix_output_dir: Optional[Path] = Path("./confusion_matrices/ALPETNet")
    frames_directory: str = "./dataset/Grid_videos/s14/video/mpg_6000/lbic1p"

@dataclass
class Config:
    """Main configuration class that aggregates all sub-configurations.
    
    This is the top-level configuration class that contains all the configuration
    components for the ALPETNet model.
    
    Attributes:
        data (DataConfig): Dataset configuration.
        model (ModelConfig): Model architecture configuration.
        training (TrainingConfig): Training process configuration.
        validation (ValidationConfig): Validation process configuration.
        decoder (DecoderConfig): CTC decoder configuration.
        analysis (AnalysisConfig): Analysis and visualization configuration.
    """
    # Use default_factory for mutable defaults
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    def __post_init__(self):
        """Initialize derived attributes after the dataclass is instantiated."""
        # Set save directory structure based on model type and data type
        self.training.save_prefix = self.training.save_prefix / f'ALPETNet_{self.data.data_type}'

