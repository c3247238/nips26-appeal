#!/usr/bin/env python3
"""
Experiment 2: Layer-wise Evolution Patterns (H2)
Characterize how attention sink patterns evolve across layers in both architectures.
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
from scipy.stats import spearmanr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

def load_sample_data(sample_size=150):
    """Load sample data for the experiment"""
    print(f"Loading {sample_size} samples...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        texts = [text for text in dataset['text'] if text.strip() and len(text.split()) > 8]
        return texts[:sample_size]
    except:
        # Fallback data
        base_texts = [
            "The attention mechanism in neural networks allows models to focus on relevant information while processing sequences.",
            "Transformers have become the dominant architecture for natural language processing tasks in recent years.",
            "Layer-wise analysis reveals how different types of attention patterns emerge throughout the model depth.",
            "Bidirectional models like BERT show different evolution patterns compared to autoregressive models like GPT."
        ]
        return base_texts * (sample_size // 4)

def compute_attention_concentration(attention_weights):
    """Compute attention sink concentration metrics"""
    # attention_weights: (batch_size, num_heads, seq_len, seq_len)
    
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Average across heads
    avg_attention = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
    
    concentrations = []
    
    for i in range(batch_size):
        attention_matrix = avg_attention[i]  # (seq_len, seq_len)
        
        # Attention concentration on first token
        first_token_attention = attention_matrix[:, 0].sum().item()
        
        # Gini coefficient for concentration measurement
        attention_dist = attention_matrix.sum(dim=0)  # incoming attention per token
        attention_dist_sorted = torch.sort(attention_dist)[0]
        n = len(attention_dist_sorted)
        cumsum = torch.cumsum(attention_dist_sorted, dim=0)
        
        if cumsum[-1] > 0:
            gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
        else:
            gini = 0
        
        # Maximum attention concentration (highest single token attention)
        max_token_attention = attention_dist.max().item()
        
        # Entropy of attention distribution
        attention_probs = attention_dist / (attention_dist.sum() + 1e-10)
        attention_probs = attention_probs[attention_probs > 1e-10]
        entropy = -(attention_probs * torch.log(attention_probs)).sum().item()
        
        concentrations.append({
            'first_token_attention': first_token_attention,
            'gini_coefficient': gini.item(),
            'max_token_attention': max_token_attention,
            'attention_entropy': entropy,
            'sequence_length': seq_len
        })
    
    return concentrations

def extract_layer_evolution(models, texts, device, max_length=128):
    """Extract layer-wise attention evolution for both models"""
    
    gpt2_model = models['gpt2_model']
    gpt2_tokenizer = models['gpt2_tokenizer']
    bert_model = models['bert_model']
    bert_tokenizer = models['bert_tokenizer']
    
    gpt2_layer_results = {layer: [] for layer in range(12)}
    bert_layer_results = {layer: [] for layer in range(12)}
    
    batch_size = 6  # Smaller batch for layer analysis
    
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
                layer_concentrations = compute_attention_concentration(attention_weights)
                
                for sample_idx, concentration in enumerate(layer_concentrations):
                    concentration.update({
                        'model': 'gpt2',
                        'layer': layer_idx,
                        'sample_idx': i + sample_idx,
                        'text_length': len(batch_texts[sample_idx])
                    })
                    gpt2_layer_results[layer_idx].append(concentration)
        
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
                layer_concentrations = compute_attention_concentration(attention_weights)
                
                for sample_idx, concentration in enumerate(layer_concentrations):
                    concentration.update({
                        'model': 'bert',
                        'layer': layer_idx,
                        'sample_idx': i + sample_idx,
                        'text_length': len(batch_texts[sample_idx])
                    })
                    bert_layer_results[layer_idx].append(concentration)
        
        except Exception as e:
            print(f"BERT batch {i} failed: {e}")
            continue
    
    return gpt2_layer_results, bert_layer_results

def analyze_layer_evolution(gpt2_layer_results, bert_layer_results):
    """Analyze layer-wise evolution patterns"""
    
    print("\n=== Layer Evolution Analysis ===")
    
    layers = list(range(12))
    
    # Compute mean concentrations per layer
    gpt2_layer_means = []
    bert_layer_means = []
    gpt2_layer_stds = []
    bert_layer_stds = []
    
    for layer in layers:
        gpt2_values = [r['first_token_attention'] for r in gpt2_layer_results[layer]]
        bert_values = [r['first_token_attention'] for r in bert_layer_results[layer]]
        
        gpt2_layer_means.append(np.mean(gpt2_values) if gpt2_values else 0)
        bert_layer_means.append(np.mean(bert_values) if bert_values else 0)
        gpt2_layer_stds.append(np.std(gpt2_values) if gpt2_values else 0)
        bert_layer_stds.append(np.std(bert_values) if bert_values else 0)
    
    # Monotonicity analysis (Spearman correlation with layer depth)
    gpt2_monotonicity, gpt2_mono_p = spearmanr(layers, gpt2_layer_means)
    bert_monotonicity, bert_mono_p = spearmanr(layers, bert_layer_means)
    
    # Curve fitting analysis
    def fit_polynomial(x, y, degree=2):
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(np.array(x).reshape(-1, 1))
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        r2 = r2_score(y, y_pred)
        return model, r2, y_pred
    
    # Fit linear and polynomial curves
    gpt2_linear_model, gpt2_linear_r2, _ = fit_polynomial(layers, gpt2_layer_means, degree=1)
    gpt2_poly_model, gpt2_poly_r2, gpt2_poly_pred = fit_polynomial(layers, gpt2_layer_means, degree=2)
    
    bert_linear_model, bert_linear_r2, _ = fit_polynomial(layers, bert_layer_means, degree=1)
    bert_poly_model, bert_poly_r2, bert_poly_pred = fit_polynomial(layers, bert_layer_means, degree=2)
    
    # Peak detection
    gpt2_peak_layer = np.argmax(gpt2_layer_means)
    bert_peak_layer = np.argmax(bert_layer_means)
    
    # Layer-to-layer gradient analysis
    gpt2_gradients = np.diff(gpt2_layer_means)
    bert_gradients = np.diff(bert_layer_means)
    
    # Statistical tests for layer differences
    layer_comparisons = []
    for i in range(11):  # Compare adjacent layers
        gpt2_layer1 = [r['first_token_attention'] for r in gpt2_layer_results[i]]
        gpt2_layer2 = [r['first_token_attention'] for r in gpt2_layer_results[i+1]]
        
        if len(gpt2_layer1) > 0 and len(gpt2_layer2) > 0:
            t_stat, p_val = stats.ttest_rel(gpt2_layer1[:min(len(gpt2_layer1), len(gpt2_layer2))],
                                           gpt2_layer2[:min(len(gpt2_layer1), len(gpt2_layer2))])
            layer_comparisons.append({'layers': f'{i}-{i+1}', 'model': 'gpt2', 't_stat': float(t_stat), 'p_value': float(p_val)})
    
    analysis = {
        'layer_means': {
            'gpt2': gpt2_layer_means,
            'bert': bert_layer_means
        },
        'layer_stds': {
            'gpt2': gpt2_layer_stds,
            'bert': bert_layer_stds
        },
        'monotonicity': {
            'gpt2_correlation': float(gpt2_monotonicity),
            'gpt2_p_value': float(gpt2_mono_p),
            'bert_correlation': float(bert_monotonicity),
            'bert_p_value': float(bert_mono_p)
        },
        'curve_fitting': {
            'gpt2_linear_r2': float(gpt2_linear_r2),
            'gpt2_polynomial_r2': float(gpt2_poly_r2),
            'bert_linear_r2': float(bert_linear_r2),
            'bert_polynomial_r2': float(bert_poly_r2)
        },
        'peak_analysis': {
            'gpt2_peak_layer': int(gpt2_peak_layer),
            'gpt2_peak_value': float(gpt2_layer_means[gpt2_peak_layer]),
            'bert_peak_layer': int(bert_peak_layer),
            'bert_peak_value': float(bert_layer_means[bert_peak_layer])
        },
        'gradient_analysis': {
            'gpt2_mean_gradient': float(np.mean(gpt2_gradients)),
            'gpt2_gradient_std': float(np.std(gpt2_gradients)),
            'bert_mean_gradient': float(np.mean(bert_gradients)),
            'bert_gradient_std': float(np.std(bert_gradients))
        },
        'layer_comparisons': layer_comparisons[:5]  # Save first 5 comparisons
    }
    
    return analysis

def create_layer_visualizations(gpt2_layer_results, bert_layer_results, analysis):
    """Create visualizations for layer evolution analysis"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Experiment 2: Layer-wise Evolution Patterns (H2)', fontsize=16)
    
    layers = list(range(12))
    gpt2_means = analysis['layer_means']['gpt2']
    bert_means = analysis['layer_means']['bert']
    gpt2_stds = analysis['layer_stds']['gpt2']
    bert_stds = analysis['layer_stds']['bert']
    
    # 1. Layer-wise evolution with error bars
    axes[0,0].errorbar(layers, gpt2_means, yerr=gpt2_stds, 
                       fmt='o-', label='GPT-2', color='skyblue', linewidth=2, capsize=5)
    axes[0,0].errorbar(layers, bert_means, yerr=bert_stds, 
                       fmt='s-', label='BERT', color='lightcoral', linewidth=2, capsize=5)
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Mean First Token Attention')
    axes[0,0].set_title('Layer-wise Attention Evolution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Monotonicity comparison
    models = ['GPT-2', 'BERT']
    correlations = [analysis['monotonicity']['gpt2_correlation'], 
                   analysis['monotonicity']['bert_correlation']]
    colors = ['skyblue', 'lightcoral']
    
    bars = axes[0,1].bar(models, correlations, color=colors, alpha=0.7)
    axes[0,1].set_ylabel('Spearman Correlation with Layer')
    axes[0,1].set_title('Monotonicity Analysis')
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{corr:.3f}', ha='center', va='bottom')
    
    # 3. R² comparison for curve fitting
    models_fit = ['GPT-2\nLinear', 'GPT-2\nPolynomial', 'BERT\nLinear', 'BERT\nPolynomial']
    r2_values = [
        analysis['curve_fitting']['gpt2_linear_r2'],
        analysis['curve_fitting']['gpt2_polynomial_r2'],
        analysis['curve_fitting']['bert_linear_r2'],
        analysis['curve_fitting']['bert_polynomial_r2']
    ]
    colors_fit = ['lightblue', 'skyblue', 'mistyrose', 'lightcoral']
    
    axes[0,2].bar(models_fit, r2_values, color=colors_fit, alpha=0.7)
    axes[0,2].set_ylabel('R² Score')
    axes[0,2].set_title('Curve Fitting Quality')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # 4. Layer gradients (changes between adjacent layers)
    gradient_layers = list(range(11))
    gpt2_gradients = np.diff(gpt2_means)
    bert_gradients = np.diff(bert_means)
    
    axes[1,0].plot(gradient_layers, gpt2_gradients, 'o-', label='GPT-2', color='skyblue', linewidth=2)
    axes[1,0].plot(gradient_layers, bert_gradients, 's-', label='BERT', color='lightcoral', linewidth=2)
    axes[1,0].set_xlabel('Layer Transition (i → i+1)')
    axes[1,0].set_ylabel('Attention Change')
    axes[1,0].set_title('Layer-to-Layer Changes')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 5. Distribution of attention values across layers
    gpt2_all_values = []
    bert_all_values = []
    layer_labels = []
    
    for layer in range(0, 12, 3):  # Sample every 3rd layer
        gpt2_values = [r['first_token_attention'] for r in gpt2_layer_results[layer]]
        bert_values = [r['first_token_attention'] for r in bert_layer_results[layer]]
        
        if gpt2_values:
            gpt2_all_values.extend(gpt2_values)
            layer_labels.extend([f'GPT2-L{layer}'] * len(gpt2_values))
        
        if bert_values:
            bert_all_values.extend(bert_values)
            layer_labels.extend([f'BERT-L{layer}'] * len(bert_values))
    
    # Create boxplot data
    box_data = []
    box_labels = []
    for layer in [0, 3, 6, 9]:
        gpt2_vals = [r['first_token_attention'] for r in gpt2_layer_results[layer]]
        bert_vals = [r['first_token_attention'] for r in bert_layer_results[layer]]
        
        if gpt2_vals:
            box_data.append(gpt2_vals)
            box_labels.append(f'GPT2-L{layer}')
        if bert_vals:
            box_data.append(bert_vals)
            box_labels.append(f'BERT-L{layer}')
    
    if box_data:
        bp = axes[1,1].boxplot(box_data, labels=box_labels)
        axes[1,1].set_ylabel('First Token Attention')
        axes[1,1].set_title('Attention Distribution by Layer')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Peak analysis
    peak_info = [
        ('GPT-2', analysis['peak_analysis']['gpt2_peak_layer'], analysis['peak_analysis']['gpt2_peak_value']),
        ('BERT', analysis['peak_analysis']['bert_peak_layer'], analysis['peak_analysis']['bert_peak_value'])
    ]
    
    for i, (model, peak_layer, peak_value) in enumerate(peak_info):
        color = 'skyblue' if model == 'GPT-2' else 'lightcoral'
        axes[1,2].bar(model, peak_layer, color=color, alpha=0.7, label=f'Peak Layer')
        
        # Add text annotation
        axes[1,2].text(i, peak_layer + 0.1, f'Layer {peak_layer}\n({peak_value:.1f}%)', 
                       ha='center', va='bottom')
    
    axes[1,2].set_ylabel('Peak Layer')
    axes[1,2].set_title('Peak Attention Layer')
    axes[1,2].set_ylim(0, 12)
    
    plt.tight_layout()
    plt.savefig('writing/figures/experiment_2_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Layer evolution visualizations saved to writing/figures/experiment_2_results.png")

def main():
    print("=== Experiment 2: Layer-wise Evolution Patterns (H2) ===")
    
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    models = load_models(device)
    
    # Load data
    texts = load_sample_data(sample_size=80)  # Smaller sample for detailed layer analysis
    
    # Extract layer-wise evolution
    print("\nExtracting layer-wise attention patterns...")
    gpt2_layer_results, bert_layer_results = extract_layer_evolution(models, texts, device)
    
    total_gpt2 = sum(len(results) for results in gpt2_layer_results.values())
    total_bert = sum(len(results) for results in bert_layer_results.values())
    print(f"Collected {total_gpt2} GPT-2 layer samples and {total_bert} BERT layer samples")
    
    # Analyze evolution patterns
    analysis = analyze_layer_evolution(gpt2_layer_results, bert_layer_results)
    
    # Print key findings
    print("\n=== Key Findings ===")
    print(f"GPT-2 monotonicity (Spearman r): {analysis['monotonicity']['gpt2_correlation']:.3f} (p={analysis['monotonicity']['gpt2_p_value']:.6f})")
    print(f"BERT monotonicity (Spearman r): {analysis['monotonicity']['bert_correlation']:.3f} (p={analysis['monotonicity']['bert_p_value']:.6f})")
    print(f"GPT-2 peak attention layer: {analysis['peak_analysis']['gpt2_peak_layer']} ({analysis['peak_analysis']['gpt2_peak_value']:.1f}%)")
    print(f"BERT peak attention layer: {analysis['peak_analysis']['bert_peak_layer']} ({analysis['peak_analysis']['bert_peak_value']:.1f}%)")
    print(f"GPT-2 linear fit R²: {analysis['curve_fitting']['gpt2_linear_r2']:.3f}")
    print(f"BERT linear fit R²: {analysis['curve_fitting']['bert_linear_r2']:.3f}")
    
    # Create visualizations
    print("\nCreating layer evolution visualizations...")
    create_layer_visualizations(gpt2_layer_results, bert_layer_results, analysis)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'layer_wise_evolution_patterns',
        'hypothesis': 'H2',
        'device': str(device),
        'sample_size': len(texts),
        'analysis': analysis,
        'summary': {
            'gpt2_samples': total_gpt2,
            'bert_samples': total_bert,
            'supports_monotonicity_hypothesis': analysis['monotonicity']['gpt2_correlation'] > analysis['monotonicity']['bert_correlation']
        },
        'status': 'success'
    }
    
    with open('exp/results/experiment_2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Experiment 2 completed successfully!")
    print("✓ Results saved to exp/results/experiment_2.json")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"❌ Experiment 2 failed: {str(e)}")
        
        # Save error information
        error_results = {
            'timestamp': datetime.now().isoformat(),
            'experiment': 'layer_wise_evolution_patterns',
            'status': 'error',
            'error': str(e)
        }
        
        with open('exp/results/experiment_2.json', 'w') as f:
            json.dump(error_results, f, indent=2)
        
        raise
