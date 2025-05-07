# decoder_beam_width.py
"""
Beam width comparative analysis script for lip reading models.

This script evaluates the performance of different beam search configurations 
for CTC decoding in lip reading models. It tests various beam width values
with different language model (LM) modes, recording WER (Word Error Rate),
CER (Character Error Rate), BLEU scores, and decoding times. Results are
saved as JSON and visualized through automatically generated plots.
"""
import torch
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from dataset_loader.lip_dataset import LipDataset
from models.ALPETNet import ALPETNet
from torchaudio.models.decoder import ctc_decoder
from dataset_loader.lip_dataset import LipDataset

# Import your config
from config.base_config import Config
from config.experiment_config import get_config

def test_beam_widths(config, beam_widths=[1, 2, 4, 8, 16],
                     lm_modes=["pure_beam", "char_lm"], output_dir=None):
    """Test different beam widths for CTC decoding using the existing config.
    
    Args:
        config: Configuration object containing model and decoder settings.
        beam_widths (list): List of beam width values to test.
        lm_modes (list): Decoding strategies to evaluate (pure_beam or char_lm).
        output_dir (str): Directory to save results and visualizations.
    
    Returns:
        dict: Results dictionary with metrics for each configuration.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a log file with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f"beam_width_log_{timestamp}.txt")
    log_file = open(log_file_path, "w")
    
    # Function to log messages to both console and file
    def log_message(message):
        """Print message to console and write to log file.
        
        Args:
            message (str): The message to log.
        """
        print(message)
        log_file.write(message + "\n")
        log_file.flush()  # Ensure logs are written immediately
    
    # Path to your model checkpoint
    model_path = config.model.pretrained_path
    
    # Load model checkpoint
    device = torch.device(f"cuda:{config.training.gpu}" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    log_message(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Initialize model with dropout settings from config
    model = ALPETNet(
        dropout_p=config.model.cnn_dropout,
        transformer_dropout=config.model.trans_dropout
    )
    
    # Load state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) # Move model to appropriate device
    
    # Create validation data loader
    val_dataset = LipDataset(config, phase='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False, # Don't shuffle for consistent evaluation
        num_workers=config.training.num_workers,
        pin_memory=True # Speed up data transfer to GPU
    )
    
    # Dictionary to store all results
    all_results = {}
    
    # Test each LM mode
    for lm_mode in lm_modes:
        log_message(f"\nTesting {lm_mode} with different beam widths:")
        
        # Skip if not a valid mode
        if lm_mode not in ["pure_beam", "char_lm"]:
            log_message(f"Invalid mode: {lm_mode}, skipping...")
            continue
        
        # Dictionary to store results for this LM mode
        mode_results = {}
        
        # Test each beam width
        for beam_width in beam_widths:
            log_message(f"  Testing beam width = {beam_width}")
            start_time = time.time()
            
            # Set decoder parameters in config dynamically
            config.decoder.mode = lm_mode
            config.decoder.beam_width = beam_width
            
            # Run validation directly without trainer logic
            wer, cer, bleu, elapsed_time = evaluate_model(model, val_loader, config, device)
            
            # Store results for this configuration
            mode_results[beam_width] = {
                "wer": wer,
                "cer": cer,
                "bleu": bleu,
                "time": elapsed_time,
                "alpha": config.decoder.alpha,
                "beta": config.decoder.beta
            }
            
            log_message(f"    WER: {wer:.4f}, CER: {cer:.4f}, BLEU: {bleu:.4f}, Time: {elapsed_time:.2f}s")
        
        all_results[lm_mode] = mode_results
    
    # Save results to JSON file
    results_file = os.path.join(output_dir, "beam_width_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    log_message(f"\nResults saved to {results_file}")
    
    # Close the log file
    log_file.close()
    
    # Generate plots for visualization
    plot_results(all_results, output_dir)
    
    return all_results

def evaluate_model(model, data_loader, config, device):
    """Evaluate the model on the validation dataset with specific decoder settings.
    
    Args:
        model: The neural network model to evaluate.
        data_loader: DataLoader containing validation samples.
        config: Configuration object with decoder parameters.
        device: Device to run evaluation on (CPU/GPU).
    
    Returns:
        tuple: (wer, cer, bleu, elapsed_time) - Average error rates and processing time.
    """
    model.eval()  # Set model to evaluation mode
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    
    with torch.no_grad(): # Disable gradient computation for inference
        for batch in data_loader:
            videos = batch['vid'].to(device)  # Move video tensors to GPU/CPU
            
            # Forward pass through the model to get logits/emissions
            outputs = model(videos)
            
            # All modes now use beam search (pure_beam or char_lm)
            log_probs = outputs.log_softmax(-1).cpu()

            # Set input lengths to sequence length for all samples in batch
            input_lengths = torch.full((log_probs.size(0),), log_probs.size(1), dtype=torch.int32)
            
            # Decode using CTC beam search
            decoder_hypotheses, tokens = decode_with_ctc(log_probs, input_lengths, config)
            
            predictions = []
            for hypothesis in decoder_hypotheses:
                if len(hypothesis) > 0:
                    best_hyp = hypothesis[0]
                    # For both pure_beam and char_lm, we use token indices
                    text = ''.join([tokens[idx] for idx in best_hyp.tokens if idx > 0])
                    predictions.append(text)
                else:
                    predictions.append("") # Empty prediction if no hypothesis
            
            # Get ground truth texts from batch
            ground_truths = [LipDataset.arr2txt(batch['txt'][i].cpu().numpy(), start=1) 
                            for i in range(len(batch['txt']))]
            
            # Collect all predictions and ground truths
            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truths)
    
    elapsed_time = time.time() - start_time
    
    # Calculate evaluation metrics
    wer_scores = LipDataset.wer(all_predictions, all_ground_truth)  # Word Error Rate
    cer_scores = LipDataset.cer(all_predictions, all_ground_truth)  # Character Error Rate
    bleu_scores = LipDataset.bleu(all_predictions, all_ground_truth)  # BLEU score
    
    return np.mean(wer_scores), np.mean(cer_scores), np.mean(bleu_scores), elapsed_time

def decode_with_ctc(log_probs, input_lengths, config):
    """Decode logits using CTC Beam Search based on configuration.
    
    Args:
        log_probs: Log probabilities from model output.
        input_lengths: Length of each sequence in the batch.
        config: Configuration with decoder parameters.
    
    Returns:
        tuple: (decoder_hypotheses, tokens) - Decoded hypotheses and token list.
    """   
    # Include the blank token in your tokens list
    tokens = [config.decoder.blank_token] + LipDataset.letters
    
    # Set up lexicon (None for pure_beam)
    lexicon = None
    
    # Choose the appropriate LM file based on mode
    lm_file = None if config.decoder.mode == "pure_beam" else config.decoder.char_lm_file_path
    
    # Initialize decoder with appropriate parameters
    decoder = ctc_decoder(
        lexicon=lexicon,
        tokens=tokens,
        lm=lm_file,
        beam_size=config.decoder.beam_width,
        lm_weight=config.decoder.alpha,  # Language model weight
        word_score=config.decoder.beta,  # Word insertion bonus
        blank_token=config.decoder.blank_token,
        sil_token=config.decoder.blank_token,  # Silence token same as blank
    )
    
    return decoder(log_probs, input_lengths), tokens

def plot_results(results, output_dir):
    """Generate plots visualizing decoding performance metrics.
    
    Creates separate graphs for each decoding strategy showing WER and CER,
    plus a graph comparing decoding times across strategies.
    
    Args:
        results (dict): Results dictionary with metrics for each configuration.
        output_dir (str): Directory to save the generated plots.
    """   
    # Create separate graph for each strategy (pure_beam and char_lm)
    for strategy, strategy_results in results.items():
        if not strategy_results:
            continue
            
        # Get only the beam widths that exist for this strategy
        beam_widths = sorted([int(bw) for bw in strategy_results.keys()])
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Set the positions of the grouped bars
        x = np.arange(len(beam_widths))
        width = 0.35
        
        # Extract metrics for available beam widths
        wer_values = [strategy_results[bw]["wer"] for bw in beam_widths]
        cer_values = [strategy_results[bw]["cer"] for bw in beam_widths]
        
        # Create the grouped bars - only WER and CER
        plt.bar(x - width/2, wer_values, width, label='WER', color='blue')
        plt.bar(x + width/2, cer_values, width, label='CER', color='orange')
        
        # Add labels, title and legend
        plt.xlabel('Beam Width', fontsize=12)
        plt.ylabel('Error Rate', fontsize=12)
        
        # Set strategy name for title
        strategy_name = "Pure Beam Search" if strategy == "pure_beam" else "Character LM Beam Search"
        plt.title(f'{strategy_name} - Error Rates Across Beam Widths', fontsize=14)
        
        # Set x-tick labels to include "W=" prefix
        plt.xticks(x, [f"W={bw}" for bw in beam_widths])
        
        # Add grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)       
        plt.legend(fontsize=10)
        plt.tight_layout()

        # Save figure to output directory
        plt.savefig(os.path.join(output_dir, f'{strategy}_error_rates.png'), dpi=300)
        plt.close()
    
    # Create the time comparison chart
    plt.figure(figsize=(12, 6))
    
    # Get all beam widths across all strategies
    all_beam_widths = set()
    for mode_results in results.values():
        all_beam_widths.update(int(bw) for bw in mode_results.keys())
    all_beam_widths = sorted(all_beam_widths)
    
    # Group by beam width
    strategies = list(results.keys())
    strategy_names = {"pure_beam": "Pure Beam Search", "char_lm": "Character LM"}
    
    # Bar positions
    x = np.arange(len(all_beam_widths))
    width = 0.8 / len(strategies)
    
    # Plot bars for each strategy
    for i, strategy in enumerate(strategies):
        if strategy not in results:
            continue

        # Collect decoding times for this strategy
        times = []
        for bw in all_beam_widths:
            if bw in results[strategy]:
                times.append(results[strategy][bw]["time"])
            else:
                times.append(0) # No data for this beam width
        
        # Calculate bar position offset
        offset = (i - len(strategies)/2 + 0.5) * width
        plt.bar(x + offset, times, width, label=strategy_names.get(strategy, strategy))
    
    # Update x-tick labels to include "W=" prefix
    plt.xlabel('Beam Width', fontsize=12)
    plt.ylabel('Decoding Time (seconds)', fontsize=12)
    plt.title('Decoding Time Comparison', fontsize=14)
    
    # Update x-tick labels to include "W=" prefix
    plt.xticks(x, [f"W={bw}" for bw in all_beam_widths])   
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)   
    plt.tight_layout()

    # Save time comparison chart
    plt.savefig(os.path.join(output_dir, 'decoding_time_comparison.png'), dpi=300)
    plt.close()

def main():
    """Main function to run the beam width comparative_analysis.
    
    Creates configuration, defines test parameters, and executes the study.
    """
    # Create config using get_config()
    config = get_config()
    
    # Define beam widths to test (adjusted for GRID corpus)
    beam_widths = [1, 2, 4, 8, 16]
    
    # Define LM modes to test - removed greedy, kept pure_beam and char_lm
    lm_modes = ["pure_beam", "char_lm"]
    
    # Run the beam width comparative_analysis
    test_beam_widths(
        config=config,
        beam_widths=beam_widths,
        lm_modes=lm_modes,
        output_dir=config.analysis.decoder_output_dir
    )

if __name__ == "__main__":
    main()
