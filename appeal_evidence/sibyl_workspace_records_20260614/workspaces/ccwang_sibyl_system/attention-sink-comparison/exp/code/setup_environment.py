#!/usr/bin/env python3
"""Setup and Environment Validation"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer
import matplotlib.pyplot as plt

def main():
    print("=== Environment Setup ===")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test model loading
    print("\nLoading models...")
    try:
        # Load GPT-2
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
        
        # Load BERT
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        
        print("✓ Models loaded successfully!")
        
        # Test attention extraction
        test_text = "The attention mechanism in transformers is fascinating."
        
        # GPT-2 test
        gpt2_inputs = gpt2_tokenizer(test_text, return_tensors="pt", max_length=20, truncation=True)
        with torch.no_grad():
            gpt2_outputs = gpt2_model(**gpt2_inputs)
            gpt2_attentions = gpt2_outputs.attentions
        
        print(f"✓ GPT-2 attention extraction: {len(gpt2_attentions)} layers")
        
        # BERT test
        bert_inputs = bert_tokenizer(test_text, return_tensors="pt", max_length=20, truncation=True)
        with torch.no_grad():
            bert_outputs = bert_model(**bert_inputs)
            bert_attentions = bert_outputs.attentions
            
        print(f"✓ BERT attention extraction: {len(bert_attentions)} layers")
        
        # Create simple validation plot
        os.makedirs('writing/figures', exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.bar(['GPT-2', 'BERT'], [len(gpt2_attentions), len(bert_attentions)])
        plt.title('Model Layer Comparison')
        plt.ylabel('Number of Layers')
        plt.savefig('writing/figures/baseline_validation.png')
        plt.close()
        print("✓ Baseline visualization created")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'pytorch_version': torch.__version__,
            'gpt2_layers': len(gpt2_attentions),
            'bert_layers': len(bert_attentions),
            'status': 'success'
        }
        
        os.makedirs('exp/results', exist_ok=True)
        with open('exp/results/setup_environment.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✓ Setup completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }
        
        os.makedirs('exp/results', exist_ok=True)
        with open('exp/results/setup_environment.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        raise

if __name__ == "__main__":
    main()
