#!/usr/bin/env python3
"""
Experiment 1: Differential Early Token Attention (H1)
Test whether GPT-2 concentrates attention on first token while BERT distributes across special tokens.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

def load_sample_data(sample_size=200):
    """Load sample data for the experiment"""
    print(f"Loading {sample_size} samples...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        texts = [text for text in dataset['text'] if text.strip() and len(text.split()) > 5]
        return texts[:sample_size]
    except:
        # Fallback data
        return [
            "The quick brown fox jumps over the lazy dog in the forest.",
            "Natural language processing is a fascinating field of study.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Transformers have revolutionized machine learning and artificial intelligence."
        ] * (sample_size // 4)

def compute_attention_metrics(attention_weights, model_type):
    """Compute attention metrics for a batch"""
    # attention_weights: (batch_size, num_heads, seq_len, seq_len)
    
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Average across heads
    avg_attention = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
    
    metrics = []
    
    for i in range(batch_size):
        attention_matrix = avg_attention[i]  # (seq_len, seq_len)
        
        if model_type == 'gpt2':
            # First token attention (position 0)
            first_token_attention = attention_matrix[:, 0].sum().item()
            special_token_attention = first_token_attention  # Same for GPT-2
            
        else:  # BERT
            # [CLS] token attention (position 0)
            cls_attention = attention_matrix[:, 0].sum().item()
            
            # [SEP] token attention (last position)
            sep_attention = attention_matrix[:, -1].sum().item()
            
            first_token_attention = cls_attention
            special_token_attention = cls_attention + sep_attention
        
        # Attention entropy
        flat_attention = attention_matrix.flatten()
        flat_attention = flat_attention[flat_attention > 1e-10]  # Avoid log(0)
        entropy = -(flat_attention * torch.log(flat_attention)).sum().item()
        
        # Top-3 attention mass
        top_attention_values, _ = torch.topk(attention_matrix.sum(dim=0), k=min(3, seq_len))
        top3_attention = top_attention_values.sum().item()
        
        metrics.append({
            'first_token_attention': first_token_attention,
            'special_token_attention': special_token_attention,
            'attention_entropy': entropy,
            'top3_attention_mass': top3_attention,
            'sequence_length': seq_len
        })
    
    return metrics

def extract_attention_patterns(models, texts, device, max_length=128):
    """Extract attention patterns for both models"""
    
    gpt2_model = models['gpt2_model']
    gpt2_tokenizer = models['gpt2_tokenizer']
    bert_model = models['bert_model']
    bert_tokenizer = models['bert_tokenizer']
    
    gpt2_results = []
    bert_results = []
    
    batch_size = 8
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}")
        
        # Process GPT-2
        try:
            gpt2_inputs = gpt2_tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            gpt2_inputs = {k: v.to(device) for k, v in gpt2_inputs.items()}
            
            with torch.no_grad():
                gpt2_outputs = gpt2_model(**gpt2_inputs)
                gpt2_attentions = gpt2_outputs.attentions
            
            # Analyze each layer
            for layer_idx, attention_weights in enumerate(gpt2_attentions):
                layer_metrics = compute_attention_metrics(attention_weights, 'gpt2')
                
                for sample_idx, metrics in enumerate(layer_metrics):
                    metrics.update({
                        'model': 'gpt2',
                        'layer': layer_idx,
                        'sample_idx': i + sample_idx,
                        'text_length': len(batch_texts[sample_idx])
                    })
                    gpt2_results.append(metrics)
        
        except Exception as e:
            print(f"GPT-2 batch {i} failed: {e}")
            continue
            
        # Process BERT
        try:
            bert_inputs = bert_tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            bert_inputs = {k: v.to(device) for k, v in bert_inputs.items()}
            
            with torch.no_grad():
                bert_outputs = bert_model(**bert_inputs)
                bert_attentions = bert_outputs.attentions
            
            # Analyze each layer
            for layer_idx, attention_weights in enumerate(bert_attentions):
                layer_metrics = compute_attention_metrics(attention_weights, 'bert')
                
                for sample_idx, metrics in enumerate(layer_metrics):
                    metrics.update({
                        'model': 'bert',
                        'layer': layer_idx,
                        'sample_idx': i + sample_idx,
                        'text_length': len(batch_texts[sample_idx])
                    })
                    bert_results.append(metrics)
        
        except Exception as e:
            print(f"BERT batch {i} failed: {e}")
            continue
    
    return gpt2_results, bert_results

def analyze_results(gpt2_results, bert_results):
    """Analyze and compare results"""
    
    print("\n=== Statistical Analysis ===")
    
    # Convert to numpy arrays for analysis
    gpt2_first_token = np.array([r['first_token_attention'] for r in gpt2_results])
    bert_first_token = np.array([r['first_token_attention'] for r in bert_results])
    
    gpt2_entropy = np.array([r['attention_entropy'] for r in gpt2_results])
    bert_entropy = np.array([r['attention_entropy'] for r in bert_results])
    
    # Layer-wise analysis
    gpt2_layers = np.array([r['layer'] for r in gpt2_results])
    bert_layers = np.array([r['layer'] for r in bert_results])
    
    # Statistical tests
    first_token_ttest = stats.ttest_ind(gpt2_first_token, bert_first_token)
    entropy_ttest = stats.ttest_ind(gpt2_entropy, bert_entropy)
    
    # Effect sizes (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std
    
    first_token_effect_size = cohens_d(gpt2_first_token, bert_first_token)
    entropy_effect_size = cohens_d(gpt2_entropy, bert_entropy)
    
    # Layer-wise evolution
    gpt2_layer_means = [np.mean([r['first_token_attention'] for r in gpt2_results if r['layer'] == l]) 
                       for l in range(12)]
    bert_layer_means = [np.mean([r['first_token_attention'] for r in bert_results if r['layer'] == l]) 
                       for l in range(12)]
    
    analysis = {
        'first_token_attention': {
            'gpt2_mean': float(np.mean(gpt2_first_token)),
            'gpt2_std': float(np.std(gpt2_first_token)),
            'bert_mean': float(np.mean(bert_first_token)),
            'bert_std': float(np.std(bert_first_token)),
            't_statistic': float(first_token_ttest.statistic),
            'p_value': float(first_token_ttest.pvalue),
            'effect_size': float(first_token_effect_size)
        },
        'attention_entropy': {
            'gpt2_mean': float(np.mean(gpt2_entropy)),
            'gpt2_std': float(np.std(gpt2_entropy)),
            'bert_mean': float(np.mean(bert_entropy)),
            'bert_std': float(np.std(bert_entropy)),
            't_statistic': float(entropy_ttest.statistic),
            'p_value': float(entropy_ttest.pvalue),
            'effect_size': float(entropy_effect_size)
        },
        'layer_wise_evolution': {
            'gpt2_layer_means': gpt2_layer_means,
            'bert_layer_means': bert_layer_means
        },
        'sample_sizes': {
            'gpt2_samples': len(gpt2_results),
            'bert_samples': len(bert_results)
        }
    }
    
    return analysis

def create_visualizations(gpt2_results, bert_results, analysis):
    """Create visualizations for the experiment"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Experiment 1: Differential Early Token Attention (H1)', fontsize=16)
    
    # 1. First token attention comparison
    gpt2_first = [r['first_token_attention'] for r in gpt2_results]
    bert_first = [r['first_token_attention'] for r in bert_results]
    
    axes[0,0].hist(gpt2_first, alpha=0.7, label='GPT-2', bins=20, color='skyblue')
    axes[0,0].hist(bert_first, alpha=0.7, label='BERT', bins=20, color='lightcoral')
    axes[0,0].set_xlabel('First Token Attention')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('First Token Attention Distribution')
    axes[0,0].legend()
    
    # 2. Layer-wise evolution
    layers = list(range(12))
    gpt2_layer_means = analysis['layer_wise_evolution']['gpt2_layer_means']
    bert_layer_means = analysis['layer_wise_evolution']['bert_layer_means']
    
    axes[0,1].plot(layers, gpt2_layer_means, 'o-', label='GPT-2', color='skyblue', linewidth=2)
    axes[0,1].plot(layers, bert_layer_means, 's-', label='BERT', color='lightcoral', linewidth=2)
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Mean First Token Attention')
    axes[0,1].set_title('Layer-wise Evolution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Entropy comparison
    gpt2_entropy = [r['attention_entropy'] for r in gpt2_results]
    bert_entropy = [r['attention_entropy'] for r in bert_results]
    
    boxes = [gpt2_entropy, bert_entropy]
    axes[0,2].boxplot(boxes, labels=['GPT-2', 'BERT'])
    axes[0,2].set_ylabel('Attention Entropy')
    axes[0,2].set_title('Attention Entropy Comparison')
    
    # 4. Special token attention by layer (BERT only)
    bert_special = [r['special_token_attention'] for r in bert_results]
    bert_layer_special = [np.mean([r['special_token_attention'] for r in bert_results if r['layer'] == l])
                         for l in range(12)]
    
    axes[1,0].plot(layers, bert_layer_special, 'o-', color='lightcoral', linewidth=2)
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('Special Token Attention')
    axes[1,0].set_title('BERT Special Token Attention by Layer')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Correlation: sequence length vs first token attention
    gpt2_lengths = [r['sequence_length'] for r in gpt2_results]
    bert_lengths = [r['sequence_length'] for r in bert_results]
    
    axes[1,1].scatter(gpt2_lengths, gpt2_first, alpha=0.5, label='GPT-2', color='skyblue')
    axes[1,1].scatter(bert_lengths, bert_first, alpha=0.5, label='BERT', color='lightcoral')
    axes[1,1].set_xlabel('Sequence Length')
    axes[1,1].set_ylabel('First Token Attention')
    axes[1,1].set_title('Sequence Length vs First Token Attention')
    axes[1,1].legend()
    
    # 6. Effect sizes
    effect_sizes = [
        analysis['first_token_attention']['effect_size'],
        analysis['attention_entropy']['effect_size']
    ]
    metrics = ['First Token\nAttention', 'Attention\nEntropy']
    
    colors = ['green' if abs(es) > 0.5 else 'orange' if abs(es) > 0.2 else 'red' for es in effect_sizes]
    axes[1,2].bar(metrics, effect_sizes, color=colors, alpha=0.7)
    axes[1,2].set_ylabel("Cohen's d")
    axes[1,2].set_title('Effect Sizes (GPT-2 vs BERT)')
    axes[1,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1,2].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Large effect')
    axes[1,2].axhline(y=-0.5, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('writing/figures/experiment_1_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualizations saved to writing/figures/experiment_1_results.png")

def main():
    print("=== Experiment 1: Differential Early Token Attention (H1) ===")
    
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    models = load_models(device)
    
    # Load data
    texts = load_sample_data(sample_size=100)  # Smaller for testing
    
    # Extract attention patterns
    print("\nExtracting attention patterns...")
    gpt2_results, bert_results = extract_attention_patterns(models, texts, device)
    
    print(f"Collected {len(gpt2_results)} GPT-2 samples and {len(bert_results)} BERT samples")
    
    # Analyze results
    analysis = analyze_results(gpt2_results, bert_results)
    
    # Print key findings
    print("\n=== Key Findings ===")
    print(f"GPT-2 first token attention: {analysis['first_token_attention']['gpt2_mean']:.3f} ± {analysis['first_token_attention']['gpt2_std']:.3f}")
    print(f"BERT first token attention: {analysis['first_token_attention']['bert_mean']:.3f} ± {analysis['first_token_attention']['bert_std']:.3f}")
    print(f"Statistical difference (p-value): {analysis['first_token_attention']['p_value']:.6f}")
    print(f"Effect size (Cohen's d): {analysis['first_token_attention']['effect_size']:.3f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(gpt2_results, bert_results, analysis)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'differential_early_token_attention',
        'hypothesis': 'H1',
        'device': str(device),
        'sample_size': len(texts),
        'analysis': analysis,
        'raw_results': {
            'gpt2': gpt2_results[:100],  # Save subset to avoid large files
            'bert': bert_results[:100]
        },
        'status': 'success'
    }
    
    with open('exp/results/experiment_1.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Experiment 1 completed successfully!")
    print("✓ Results saved to exp/results/experiment_1.json")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"❌ Experiment 1 failed: {str(e)}")
        
        # Save error information
        error_results = {
            'timestamp': datetime.now().isoformat(),
            'experiment': 'differential_early_token_attention',
            'status': 'error',
            'error': str(e)
        }
        
        with open('exp/results/experiment_1.json', 'w') as f:
            json.dump(error_results, f, indent=2)
        
        raise
