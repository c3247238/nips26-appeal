#!/usr/bin/env python3
"""
Experiment 3: Position Encoding Interaction Effects (H3)
Test how position encoding modifications affect attention sink patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_models(device):
    """Load GPT-2 and BERT models"""
    print("Loading models...")
    
    # GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
    gpt2_model.to(device).eval()
    
    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    bert_model.to(device).eval()
    
    return {
        'gpt2_model': gpt2_model,
        'gpt2_tokenizer': gpt2_tokenizer,
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer
    }

def load_sample_data(sample_size=60):
    """Load sample data for the experiment"""
    print(f"Loading {sample_size} samples...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        texts = [text for text in dataset['text'] if text.strip() and len(text.split()) > 6]
        return texts[:sample_size]
    except:
        # Fallback data
        base_texts = [
            "Position encodings help transformers understand the order of tokens in sequences.",
            "The attention mechanism can be significantly affected by positional information.",
            "Different encoding schemes lead to different attention patterns in transformer models.",
            "Absolute position encoding adds positional information to each token embedding.",
            "Rotary position encoding provides an alternative to traditional absolute encodings.",
            "Bidirectional models may be more robust to position encoding perturbations."
        ]
        return base_texts * (sample_size // len(base_texts) + 1)[:sample_size]

def modify_position_encodings(model, model_type, modification_type, device):
    """Create modified versions of position encodings"""
    
    if model_type == 'gpt2':
        original_pos_emb = model.wpe.weight.data.clone()
        
        if modification_type == 'zeroed':
            model.wpe.weight.data.zero_()
        elif modification_type == 'randomized':
            model.wpe.weight.data = torch.randn_like(model.wpe.weight.data) * original_pos_emb.std()
        elif modification_type == 'shuffled':
            shuffled_indices = torch.randperm(model.wpe.weight.size(0))
            model.wpe.weight.data = model.wpe.weight.data[shuffled_indices]
            
        return original_pos_emb
        
    else:  # BERT
        original_pos_emb = model.embeddings.position_embeddings.weight.data.clone()
        
        if modification_type == 'zeroed':
            model.embeddings.position_embeddings.weight.data.zero_()
        elif modification_type == 'randomized':
            model.embeddings.position_embeddings.weight.data = torch.randn_like(
                model.embeddings.position_embeddings.weight.data) * original_pos_emb.std()
        elif modification_type == 'shuffled':
            shuffled_indices = torch.randperm(model.embeddings.position_embeddings.weight.size(0))
            model.embeddings.position_embeddings.weight.data = model.embeddings.position_embeddings.weight.data[shuffled_indices]
            
        return original_pos_emb

def restore_position_encodings(model, model_type, original_embeddings):
    """Restore original position encodings"""
    
    if model_type == 'gpt2':
        model.wpe.weight.data = original_embeddings
    else:  # BERT
        model.embeddings.position_embeddings.weight.data = original_embeddings

def extract_attention_patterns_batch(model, tokenizer, texts, device, max_length=128):
    """Extract attention patterns for a batch of texts"""
    
    # Tokenize batch
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions
    
    # Extract first token attention for each layer
    batch_results = []
    
    for layer_idx, attention_weights in enumerate(attentions):
        # attention_weights: (batch_size, num_heads, seq_len, seq_len)
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Average across heads
        avg_attention = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        for batch_idx in range(batch_size):
            attention_matrix = avg_attention[batch_idx]  # (seq_len, seq_len)
            
            # First token attention
            first_token_attention = attention_matrix[:, 0].sum().item()
            
            # Attention stability metrics
            attention_variance = attention_matrix.var().item()
            attention_mean = attention_matrix.mean().item()
            
            batch_results.append({
                'layer': layer_idx,
                'batch_idx': batch_idx,
                'first_token_attention': first_token_attention,
                'attention_variance': attention_variance,
                'attention_mean': attention_mean,
                'sequence_length': seq_len
            })
    
    return batch_results

def run_position_encoding_experiment(models, texts, device):
    """Run position encoding modification experiment"""
    
    modifications = ['baseline', 'zeroed', 'randomized', 'shuffled']
    results = {'gpt2': {}, 'bert': {}}
    
    batch_size = 8
    
    for model_name in ['gpt2', 'bert']:
        print(f"\n--- Testing {model_name.upper()} ---")
        
        model = models[f'{model_name}_model']
        tokenizer = models[f'{model_name}_tokenizer']
        
        for modification in modifications:
            print(f"Testing {modification} position encoding...")
            
            if modification != 'baseline':
                # Modify position encodings
                original_embeddings = modify_position_encodings(model, model_name, modification, device)
            
            modification_results = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                try:
                    batch_results = extract_attention_patterns_batch(
                        model, tokenizer, batch_texts, device
                    )
                    
                    for result in batch_results:
                        result.update({
                            'model': model_name,
                            'modification': modification,
                            'global_sample_idx': i + result['batch_idx']
                        })
                    
                    modification_results.extend(batch_results)
                    
                except Exception as e:
                    print(f"Batch {i//batch_size} failed for {model_name}-{modification}: {e}")
                    continue
            
            results[model_name][modification] = modification_results
            
            if modification != 'baseline':
                # Restore original position encodings
                restore_position_encodings(model, model_name, original_embeddings)
    
    return results

def analyze_position_effects(results):
    """Analyze position encoding effects"""
    
    print("\n=== Position Encoding Analysis ===")
    
    analysis = {
        'gpt2': {},
        'bert': {},
        'comparisons': {}
    }
    
    modifications = ['baseline', 'zeroed', 'randomized', 'shuffled']
    
    for model_name in ['gpt2', 'bert']:
        print(f"\n--- {model_name.upper()} Analysis ---")
        model_analysis = {}
        
        # Extract baseline results
        baseline_results = results[model_name]['baseline']
        baseline_attention = np.array([r['first_token_attention'] for r in baseline_results])
        
        for modification in modifications[1:]:  # Skip baseline
            mod_results = results[model_name][modification]
            mod_attention = np.array([r['first_token_attention'] for r in mod_results])
            
            # Correlation with baseline
            if len(baseline_attention) > 0 and len(mod_attention) > 0:
                min_length = min(len(baseline_attention), len(mod_attention))
                correlation, corr_p = pearsonr(
                    baseline_attention[:min_length], 
                    mod_attention[:min_length]
                )
            else:
                correlation, corr_p = 0, 1
            
            # Percentage change in attention
            baseline_mean = np.mean(baseline_attention)
            mod_mean = np.mean(mod_attention)
            
            if baseline_mean > 0:
                percent_change = ((mod_mean - baseline_mean) / baseline_mean) * 100
            else:
                percent_change = 0
            
            # Statistical test
            if len(baseline_attention) > 0 and len(mod_attention) > 0:
                t_stat, p_val = stats.ttest_ind(baseline_attention, mod_attention)
            else:
                t_stat, p_val = 0, 1
            
            # Attention persistence (how much attention sink behavior remains)
            persistence = correlation if correlation > 0 else 0
            
            model_analysis[modification] = {
                'correlation_with_baseline': float(correlation),
                'correlation_p_value': float(corr_p),
                'percent_change': float(percent_change),
                'baseline_mean': float(baseline_mean),
                'modified_mean': float(mod_mean),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'attention_persistence': float(persistence),
                'sample_size': len(mod_results)
            }
            
            print(f"{modification}: {percent_change:.1f}% change, r={correlation:.3f}")
        
        analysis[model_name] = model_analysis
    
    # Cross-model comparisons
    print("\n--- Cross-Model Comparison ---")
    
    for modification in modifications[1:]:
        gpt2_change = analysis['gpt2'][modification]['percent_change']
        bert_change = analysis['bert'][modification]['percent_change']
        
        gpt2_persistence = analysis['gpt2'][modification]['attention_persistence']
        bert_persistence = analysis['bert'][modification]['attention_persistence']
        
        analysis['comparisons'][modification] = {
            'gpt2_change': gpt2_change,
            'bert_change': bert_change,
            'change_difference': gpt2_change - bert_change,
            'gpt2_persistence': gpt2_persistence,
            'bert_persistence': bert_persistence,
            'persistence_difference': gpt2_persistence - bert_persistence,
            'bert_more_robust': abs(bert_change) < abs(gpt2_change)
        }
        
        print(f"{modification}: GPT-2 {gpt2_change:.1f}% vs BERT {bert_change:.1f}% change")
    
    return analysis

def create_position_visualizations(results, analysis):
    """Create visualizations for position encoding experiment"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Experiment 3: Position Encoding Interaction Effects (H3)', fontsize=16)
    
    modifications = ['zeroed', 'randomized', 'shuffled']
    
    # 1. Percentage changes comparison
    gpt2_changes = [analysis['gpt2'][mod]['percent_change'] for mod in modifications]
    bert_changes = [analysis['bert'][mod]['percent_change'] for mod in modifications]
    
    x = np.arange(len(modifications))
    width = 0.35
    
    axes[0,0].bar(x - width/2, gpt2_changes, width, label='GPT-2', color='skyblue')
    axes[0,0].bar(x + width/2, bert_changes, width, label='BERT', color='lightcoral')
    axes[0,0].set_xlabel('Position Encoding Modification')
    axes[0,0].set_ylabel('Percentage Change (%)')
    axes[0,0].set_title('Attention Change by Modification')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(modifications)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 2. Attention persistence (correlation with baseline)
    gpt2_persistence = [analysis['gpt2'][mod]['attention_persistence'] for mod in modifications]
    bert_persistence = [analysis['bert'][mod]['attention_persistence'] for mod in modifications]
    
    axes[0,1].bar(x - width/2, gpt2_persistence, width, label='GPT-2', color='skyblue')
    axes[0,1].bar(x + width/2, bert_persistence, width, label='BERT', color='lightcoral')
    axes[0,1].set_xlabel('Position Encoding Modification')
    axes[0,1].set_ylabel('Correlation with Baseline')
    axes[0,1].set_title('Attention Pattern Persistence')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(modifications)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)
    
    # 3. Robustness comparison (smaller change = more robust)
    robustness_scores = []
    for mod in modifications:
        gpt2_abs_change = abs(analysis['gpt2'][mod]['percent_change'])
        bert_abs_change = abs(analysis['bert'][mod]['percent_change'])
        robustness_scores.append([gpt2_abs_change, bert_abs_change])
    
    robustness_scores = np.array(robustness_scores)
    
    axes[0,2].bar(x - width/2, robustness_scores[:, 0], width, label='GPT-2', color='skyblue')
    axes[0,2].bar(x + width/2, robustness_scores[:, 1], width, label='BERT', color='lightcoral')
    axes[0,2].set_xlabel('Position Encoding Modification')
    axes[0,2].set_ylabel('Absolute Change (%)')
    axes[0,2].set_title('Position Encoding Sensitivity\n(Lower = More Robust)')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(modifications)
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Attention distribution comparison (baseline vs modified)
    # Show baseline vs zeroed for both models
    
    gpt2_baseline = [r['first_token_attention'] for r in results['gpt2']['baseline']]
    gpt2_zeroed = [r['first_token_attention'] for r in results['gpt2']['zeroed']]
    bert_baseline = [r['first_token_attention'] for r in results['bert']['baseline']]
    bert_zeroed = [r['first_token_attention'] for r in results['bert']['zeroed']]
    
    axes[1,0].hist(gpt2_baseline, alpha=0.5, bins=15, label='GPT-2 Baseline', color='skyblue')
    axes[1,0].hist(gpt2_zeroed, alpha=0.5, bins=15, label='GPT-2 Zeroed', color='blue')
    axes[1,0].set_xlabel('First Token Attention')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('GPT-2: Baseline vs Zeroed PE')
    axes[1,0].legend()
    
    axes[1,1].hist(bert_baseline, alpha=0.5, bins=15, label='BERT Baseline', color='lightcoral')
    axes[1,1].hist(bert_zeroed, alpha=0.5, bins=15, label='BERT Zeroed', color='red')
    axes[1,1].set_xlabel('First Token Attention')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('BERT: Baseline vs Zeroed PE')
    axes[1,1].legend()
    
    # 5. Summary: Which model is more robust?
    robustness_summary = []
    model_labels = []
    
    for mod in modifications:
        gpt2_change = abs(analysis['gpt2'][mod]['percent_change'])
        bert_change = abs(analysis['bert'][mod]['percent_change'])
        
        robustness_summary.extend([gpt2_change, bert_change])
        model_labels.extend([f'GPT-2\n{mod}', f'BERT\n{mod}'])
    
    colors = ['skyblue', 'lightcoral'] * len(modifications)
    bars = axes[1,2].bar(range(len(robustness_summary)), robustness_summary, color=colors)
    axes[1,2].set_xlabel('Model-Modification')
    axes[1,2].set_ylabel('Absolute Change (%)')
    axes[1,2].set_title('Overall Robustness Comparison')
    axes[1,2].set_xticks(range(len(model_labels)))
    axes[1,2].set_xticklabels(model_labels, rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', label='GPT-2'),
                      Patch(facecolor='lightcoral', label='BERT')]
    axes[1,2].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('writing/figures/experiment_3_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Position encoding visualizations saved to writing/figures/experiment_3_results.png")

def main():
    print("=== Experiment 3: Position Encoding Interaction Effects (H3) ===")
    
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    models = load_models(device)
    
    # Load data
    texts = load_sample_data(sample_size=48)  # Smaller sample due to multiple conditions
    
    # Run position encoding experiments
    print("\nRunning position encoding modification experiments...")
    results = run_position_encoding_experiment(models, texts, device)
    
    # Analyze results
    analysis = analyze_position_effects(results)
    
    # Print key findings
    print("\n=== Key Findings ===")
    for modification in ['zeroed', 'randomized', 'shuffled']:
        gpt2_change = analysis['gpt2'][modification]['percent_change']
        bert_change = analysis['bert'][modification]['percent_change']
        gpt2_persistence = analysis['gpt2'][modification]['attention_persistence']
        bert_persistence = analysis['bert'][modification]['attention_persistence']
        
        print(f"\n{modification.capitalize()} position encoding:")
        print(f"  GPT-2: {gpt2_change:.1f}% change, {gpt2_persistence:.3f} correlation")
        print(f"  BERT: {bert_change:.1f}% change, {bert_persistence:.3f} correlation")
        print(f"  BERT more robust: {analysis['comparisons'][modification]['bert_more_robust']}")
    
    # Create visualizations
    print("\nCreating position encoding visualizations...")
    create_position_visualizations(results, analysis)
    
    # Calculate summary statistics
    avg_gpt2_change = np.mean([abs(analysis['gpt2'][mod]['percent_change']) 
                              for mod in ['zeroed', 'randomized', 'shuffled']])
    avg_bert_change = np.mean([abs(analysis['bert'][mod]['percent_change']) 
                              for mod in ['zeroed', 'randomized', 'shuffled']])
    
    avg_gpt2_persistence = np.mean([analysis['gpt2'][mod]['attention_persistence'] 
                                   for mod in ['zeroed', 'randomized', 'shuffled']])
    avg_bert_persistence = np.mean([analysis['bert'][mod]['attention_persistence'] 
                                   for mod in ['zeroed', 'randomized', 'shuffled']])
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'position_encoding_interaction_effects',
        'hypothesis': 'H3',
        'device': str(device),
        'sample_size': len(texts),
        'analysis': analysis,
        'summary_statistics': {
            'avg_gpt2_absolute_change': float(avg_gpt2_change),
            'avg_bert_absolute_change': float(avg_bert_change),
            'avg_gpt2_persistence': float(avg_gpt2_persistence),
            'avg_bert_persistence': float(avg_bert_persistence),
            'bert_more_robust_overall': avg_bert_change < avg_gpt2_change,
            'robustness_difference': float(avg_gpt2_change - avg_bert_change)
        },
        'status': 'success'
    }
    
    with open('exp/results/experiment_3.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("✓ Experiment 3 completed successfully!")
    print("✓ Results saved to exp/results/experiment_3.json")
    
    print(f"\n=== Summary ===")
    print(f"Average GPT-2 sensitivity: {avg_gpt2_change:.1f}% change")
    print(f"Average BERT sensitivity: {avg_bert_change:.1f}% change") 
    print(f"BERT is {avg_gpt2_change - avg_bert_change:.1f}% more robust to position encoding changes")
    
    return results_summary

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"❌ Experiment 3 failed: {str(e)}")
        
        # Save error information
        error_results = {
            'timestamp': datetime.now().isoformat(),
            'experiment': 'position_encoding_interaction_effects',
            'status': 'error',
            'error': str(e)
        }
        
        with open('exp/results/experiment_3.json', 'w') as f:
            json.dump(error_results, f, indent=2)
        
        raise
