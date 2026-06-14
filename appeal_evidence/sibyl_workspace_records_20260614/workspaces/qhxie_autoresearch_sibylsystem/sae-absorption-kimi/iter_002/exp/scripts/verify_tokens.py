"""Verify that all hierarchy and control tokens are single tokens in GPT-2 and Pythia vocabularies."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

def is_single_token(tokenizer, word):
    tokens = tokenizer.encode(word, add_special_tokens=False)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True).strip().lower()
    return len(tokens) == 1 and decoded == word.strip().lower()

# Load verified tokens spec
spec_path = Path("plan/verified_tokens.json")
spec = json.loads(spec_path.read_text())

# Load tokenizers
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
pythia_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

all_tokens = set()
for h in spec["hierarchies"]:
    all_tokens.add(h["parent"])
    all_tokens.update(h["children"])
for pair in spec["control_pairs"]:
    all_tokens.update(pair)

results = {
    "gpt2": {},
    "pythia_160m": {},
    "all_single_token": {"gpt2": True, "pythia_160m": True}
}

for word in sorted(all_tokens):
    g_ok = is_single_token(gpt2_tok, word)
    p_ok = is_single_token(pythia_tok, word)
    results["gpt2"][word] = g_ok
    results["pythia_160m"][word] = p_ok
    if not g_ok:
        results["all_single_token"]["gpt2"] = False
    if not p_ok:
        results["all_single_token"]["pythia_160m"] = False

out_path = Path("exp/results/verify_tokens_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print(json.dumps(results["all_single_token"], indent=2))
