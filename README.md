# Final Year Individual Dissertation

## Description

This dissertation explores a hybrid approach to Visual Speech Recognition
(VSR), commonly known as lip reading, which is crucial for enhancing communication in environments with compromised audio, such as surveillance, noisy settings,
or for individuals with hearing impairments. The proposed system integrates convolutional and transformer-based modules to capture detailed spatial and temporal
features from silent video frames of lip movements.
Two novel architectures are developed: LPETNet (Light Patch Embedding Transformer Network) and ALPETNet (Attention-reinforced Light Patch Embedding Transformer Network). LPETNet focuses on efficiency through patch embeddings and
depthwise separable 3D convolutions, while ALPETNet enhances performance
further by incorporating gated transformers and channel attention mechanisms.
Both models fuse spatial and temporal features and refine sequence modeling
using BiGRUs.
The final system successfully balances robust performance and computational
efficiency and yields promising results for real-time, interpretable lip-reading applications. This dissertation advances the field of hybrid feature fusion in deep
learning and highlights the potential of such architectures for practical deployment
in VSR systems.

## Getting Started

### Prerequisites

*   Python 3.12.9+
*   PyTorch 2.6+
*   CUDA-compatible GPU (recommended)
*   Additional libraries listed in `requirements.txt`
*   Linux environment (required for KenLM)

### Installation

#### Option 1: Clone the repository

    
    # Clone the repository
    git clone https://github.com/Kokagi-sama/FYP.git
    
    # Navigate to the source directory
    cd FYP/src
    

#### Option 2: Use the provided ZIP file

1.  Extract the ZIP file to your desired location
2.  Navigate to the source directory:
    
        cd <path/to/extracted>/src
    

#### Setting up the environment

This project requires KenLM which primarily works in Linux environments. While Windows Subsystem for Linux (WSL) can be used, we cannot guarantee full compatibility with this setup.

    
    # Create a Python virtual environment
    python -m venv venv
    
    # Activate the virtual environment
    # On Linux/Mac:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    
    # Install PyTorch with CUDA 12.6 support first (from pytorch-cuda.txt)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    
    # Install remaining dependencies
    pip install -r requirements.txt
    

Note: If you're using Windows, you may encounter compatibility issues with KenLM. For best results, we recommend using a native Linux environment.
    

#### Installing KenLM (Linux only)

If pip doesn't successfully install KenLM through requirements.txt, you can use the installation script included in the project:

    
    # Navigate to the 5-gram-model directory
    cd 5-gram-model
    
    # Make the installation script executable
    chmod +x install_kenlm.sh
    
    # Run the installation script
    ./install_kenlm.sh
    

Alternatively, if the script doesn't work, you can manually install KenLM following these steps:

    
    # Install build dependencies
    sudo apt-get update
    sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
    
    # Clone KenLM repository
    git clone https://github.com/kpu/kenlm
    
    # Build KenLM
    cd kenlm
    mkdir -p build
    cd build
    cmake ..
    make -j 4
    sudo make install
    

Note: KenLM can be challenging to install through pip due to its build requirements and dependencies. Manual installation is often more reliable.

### Downloading and Preprocessing GRID Corpus Data

This project uses the GRID Audio-Visual Speech Corpus. You'll need to obtain and preprocess this data:

#### 1\. Download GRID Corpus

Download the data from [https://zenodo.org/records/3625687](https://zenodo.org/records/3625687):

*   **alignments.zip** - Word-level time alignments for all speakers
*   **Speaker video files** - s1.zip through s34.zip (note: s21.zip is missing)

#### 2\. Required Preprocessing

After downloading, you'll need to process the data into the correct format:

1.  **Extract alignments** - Extract `alignments.zip` and place them in the following structure:
    
    *   📁 dataset
        *   📁 Grid_alignments
            *   📁 alignments
                *   📁 [speaker, i.e. s1]
                    *   📄 [video name, i.e. bbaf2n].align 
    
2.  **Process video frames** - You'll need to extract frames from each video and organise them appropriately. The processed frames should be placed in the following structure:
    
    *   📁 dataset
        *   📁 Grid_videos
            *   📁 [speaker, i.e. s1]
                *   📁 video
                    *   📁 mpg_6000
                        *   📁 [video name, i.e. bbaf2n]
                            *  🖼️ 1.jpg
                            *  🖼️ 2.jpg
                            *  🖼️ 3.jpg
                            *   ...
                            *  🖼️ 75.jpg    
    
3.  **Facial landmark detection** - You'll need to use facial landmark detection libraries (such as dlib and face\_alignment) to identify mouth regions in each frame
4.  **Mouth region extraction** - Crop the lip regions based on detected landmarks

**Important:** This preprocessing involves:

*   Converting videos to frame sequences (75 frames per video)
*   Detecting facial landmarks using CUDA-accelerated detection
*   Extracting and normalising lip regions

You will need to implement this preprocessing pipeline yourself using libraries such as `dlib`, `face_alignment`, and `OpenCV`. The preprocessing can be time-consuming and requires significant disk space.

**Note:** For the purpose of this project, we assume the mouth regions have already been preprocessed using dlib, OpenCV, and face\_alignment's FaceAlignment class with CUDA acceleration, and are already organised in the expected directory structure mentioned above.

## Code Structure
*   📁 5-gram-model
    *   📁 character
        *   📄 grid\_char.arpa
        *   📄 grid\_char.binary
        *   📄 grid\_char.txt
        *   $  train\_lm\_char.sh
    *   📁 word
        *   📄 grid\_word.arpa
        *   📄 grid\_word.binary
        *   📄 grid\_word.txt
        *   📄 lexicon.txt
        *   $  train\_lm\_word.sh
    *   🐳  docker-compose.yml
    *   🐳  Dockerfile
    *   📄 grid.txt
    *   $  install\_kenlm.sh
*   📁 beam\_width\_comparative\_analysis
    *   📁 ALPETNet
        *   📄 beam\_width\_log\_20250331\_212414.txt
        *   {} beam\_width\_results.json
        *   🖼️ char\_lm\_error\_rates.jpg
        *   🖼️ decoding\_time\_comparison.jpg
        *   🖼️ pure\_beam\_error\_rates.jpg
    *   📁 LPETNet
        *   📄 beam\_width\_log\_20250405\_133039.txt
        *   📄 beam\_width\_results.json
        *   🖼️ char\_lm\_error\_rates.jpg
        *   🖼️ decoding\_time\_comparison.jpg
        *   🖼️ pure\_beam\_error\_rates.jpg
*   📁 config
    *   🐍 base\_config.py
    *   🐍 experiment\_config.py
*   📁 confusion\_matrices
    *   📁 ALPETNet
        *   📄 alphabet\_confusion\_matrix\_20250415\_130811.npy
        *   🖼️ alphabet\_confusion\_matrix\_20250415\_130811.png
        *   📄 phoneme\_confusion\_matrix\_20250415\_130811.npy
        *   🖼️ phoneme\_confusion\_matrix\_20250415\_130811.png
*   📁 dataset
    *   📁 data
        *   📄 overlap\_train.txt
        *   📄 overlap\_val.txt
*   📁 dataset\_loader
        *   🐍 base\_dataset.py
        *   🐍 lip\_dataset.py
        *   🐍 transforms.py
*   📁 models
    *   🐍 ALPETNet.py
    *   🐍 LPETNet.py
*   📁 plots
    *   📁 plots\_ALPETNet
    *   📁 plots\_LPETNet
*   📁 pretrain\_models
    *   📄 ALPETNet\_20250326\_042518\_loss\_0.0608\_wer\_0.0416\_cer\_0.0171\_bleu\_0.9584.pt
    *   📄 LPETNet\_20250328\_142608\_loss\_0.0630\_wer\_0.0519\_cer\_0.0214\_bleu\_0.9487.pt
*   📁 saliency\_map
    *   📁 ALPETNet
        *  🖼️ saliency\_map\_output.png
*   📁 trainer
    *   🐍 trainer\_alpetnet.py
    *   🐍 trainer\_comparative\_analysis.py
    *   🐍 trainer\_lpetnet.py
*   🐍 confusion\_matrix.py
*   🐍 decoder\_beam\_width.py
*   🐍 main\_alpetnet.py
*   🐍 main\_lpetnet.py
*   📄 pytorch\_cuda.txt
*   📄 requirements.txt
*   🐍 saliency\_map.py

## Configuration System

The project uses a comprehensive configuration system built with Python dataclasses, ensuring flexibility and maintainability across different experiments.

### Configuration Structure

The configuration is organized into specialized categories:

*   **DataConfig**: Controls dataset paths and processing parameters
    *   Specifies dataset split type ('overlap')
    *   Defines paths to video and annotation directories
    *   Sets padding lengths for sequences
*   **ModelConfig**: Defines model architecture parameters
    *   Sets dropout rates for transformer (0.1) and CNN layers (0.5)
    *   Controls model dimensions (d\_model=512) and attention heads (nhead=8)
    *   Contains paths for saving/loading model checkpoints
*   **TrainingConfig**: Manages training hyperparameters
    *   Specifies GPU device, batch size (64), and learning rate (1e-3)
    *   Sets maximum epochs (120) and display intervals
*   **DecoderConfig**: Controls CTC decoding strategies
    *   Supports multiple decoding modes: greedy, char\_lm, word\_lm
    *   Configures beam search parameters (width=10, alpha=1.0, beta=1.5)
    *   Sets paths to language model files in the 5-gram-model directory
*   **AnalysisConfig**: Defines output paths for analysis artifacts
    *   Beam width comparison results
    *   Saliency maps and confusion matrices

### Usage

    
    from config.experiment_config import get_config
    
    # Get the default configuration
    config = get_config()
    
    # Access specific settings
    batch_size = config.training.batch_size
    model_path = config.model.pretrained_path
    

### Important Notes

*   **ALPETNet Focus**: The default configuration is tailored for ALPETNet rather than LPETNet:
    *   Default pretrained model path: `./pretrain_models/ALPETNet_20250326_042518_loss_0.0608_wer_0.0416_cer_0.0171_bleu_0.9584.pt`
    *   Save directories and analysis paths are set for ALPETNet
*   To use LPETNet instead, modify the relevant paths in the configuration.

## Usage

### Training Scripts

~~~
python main_lpetnet.py
~~~

This script trains the LPETNet (Lightweight Patch-Efficient Transformer Network) model for lip reading. It processes video data through the training pipeline defined in `trainer_lpetnet.py`, applying the model architecture from `models/LPETNet.py`. The training process includes:

*   Loading the dataset with appropriate data augmentation
*   Setting up optimisation parameters (learning rate, weight decay)
*   Running training loops with periodic validation
*   Automatically saving checkpoints with performance metrics (WER, CER, BLEU)
*   Generating training progress plots in the `plots/plots_LPETNet` directory
*   Saves training weights in a new `weights/LPETNet_overlap` directory

~~~
python main_alpetnet.py
~~~

Similar to the LPETNet script, but trains the more advanced ALPETNet (Attention-enhanced LPET Network) model. This version adds attention mechanisms and improved feature extraction to the base architecture:

*   Uses channel attention and cross-attention fusion mechanisms
*   Implements multi-scale patch embedding for improved feature extraction
*   Follows the same training paradigm but applies the enhanced model architecture
*   Saves checkpoints and visualisations to the ALPETNet-specific directories
*   Saves weights in a new `weights/ALPETNet_overlap` directory

### Evaluation Scripts
~~~
python decoder_beam_width_matrix.py
~~~

This script performs beam width ablation studies to evaluate the impact of different beam search configurations on decoding performance:

*   Tests multiple beam width values (e.g., 1, 2, 4, 8, 16)
*   Evaluates performance with different language model modes (pure beam search, character-level LM)
*   Records WER, CER, BLEU scores, and processing time for each configuration
*   Generates comparative charts and saves results to the `beam_width_comparative_analysis` directory

~~~
python confusion_matrix.py
~~~

Generates confusion matrices to analyse which characters or phonemes the model confuses most frequently:

*   Runs validation data through the pretrained model
*   Creates character-level confusion matrices showing error patterns
*   Generates phoneme-level confusion matrices (using ARPABET phoneme representation)
*   Saves visualisations as heatmaps to the `confusion_matrices/ALPETNet` directory
*   Helps identify systematic errors in model predictions

### Saliency Visualisation

~~~
python saliency_map.py
~~~

Creates visual explanations showing which regions of input video frames influence the model's predictions:

*   Processes video frames through the trained model
*   Calculates gradients of output predictions with respect to input pixels
*   Generates heatmap overlays highlighting the most influential regions
*   Saves visualisations to the `saliency_map/ALPETNet` directory
*   Helps interpret how the model focuses on specific facial features during lip reading

Each script uses the configuration from `config/` and accesses models saved in the `pretrain_models/` directory.

## Acknowledgments

* GRID corpus for the dataset
* Dr. Tissa Chandesa for his guidance and valuable insights
