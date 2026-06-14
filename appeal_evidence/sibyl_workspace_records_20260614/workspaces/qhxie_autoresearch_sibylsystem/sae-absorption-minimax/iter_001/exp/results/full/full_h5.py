"""
Full H5: Comprehensive downstream task analysis.

Expands to 200 features (low/mid/high absorption) across GPT-2 L8 and GPT-2 L6.
Evaluates on 3 downstream tasks:
  - Task 1 (simple): Gender detection from names (binary)
  - Task 2 (simple): Sentiment detection (binary)
  - Task 3 (causal): Logical implication detection (causal reasoning)

Pass criteria:
  - Simple task accuracy delta < 3% across absorption levels
  - Causal task accuracy delta > 8%

Using neg_cos_sim_var as absorption proxy (best predictor from h4_uas_dev: r=-0.49).
"""

import json
import os
import gc
import warnings
import sys
import numpy as np
import torch
from scipy.stats import spearmanr
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
REMOTE_BASE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = REMOTE_BASE / "exp/results/full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "full_h5"
DEVICE = "cuda"
SEED = 42
N_FEATURES_PER_MODEL = 100
N_SAMPLES_PER_CLASS = 500
BATCH_SIZE = 64

n_features_per_bin = N_FEATURES_PER_MODEL  # per absorption bin


def log(msg):
    """Print with flush for immediate visibility."""
    print(msg, flush=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_neg_cos_sim_var(sae, feature_indices):
    """Compute neg_cos_sim_var as absorption proxy (from h4_uas_dev findings).
    Higher neg_cos_sim_var = higher absorption.
    """
    log("  Computing neg_cos_sim_var absorption proxy...")
    W_dec = sae.W_dec.detach().to("cpu").numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)

    scores = {}
    for idx, feat_idx in enumerate(feature_indices):
        if idx % 50 == 0:
            log(f"    Feature {idx}/{len(feature_indices)}")
        other_indices = [i for i in feature_indices if i != feat_idx]
        if len(other_indices) < 2:
            scores[feat_idx] = 0.0
            continue
        cos_sims = np.dot(W_dec_norm[feat_idx], W_dec_norm[other_indices].T)
        scores[feat_idx] = -float(np.var(cos_sims))
    log("  Absorption proxy computed.")
    return scores


def get_top_activated_features(activations, n_features):
    """Get top-n features by total activation magnitude."""
    total_act = torch.sum(torch.abs(activations.detach()), dim=(0, 1)).cpu().numpy()
    top_indices = np.argsort(total_act)[-n_features:][::-1].tolist()
    return top_indices


def bin_features_by_absorption(absorption_scores, n_per_bin, seed=42):
    """Bin features into low/mid/high absorption groups."""
    rng = np.random.RandomState(seed)
    sorted_features = sorted(absorption_scores.items(), key=lambda x: x[1])
    n = len(sorted_features)
    n_per = n // 3

    low = [f for f, _ in sorted_features[:n_per]]
    mid = [f for f, _ in sorted_features[n_per:2*n_per]]
    high = [f for f, _ in sorted_features[2*n_per:]]

    rng.shuffle(low); rng.shuffle(mid); rng.shuffle(high)
    return {
        "low": low[:n_per_bin],
        "mid": mid[:n_per_bin],
        "high": high[:n_per_bin],
    }


# ======================================================================
# TASK 1: Gender detection from names (simple)
# ======================================================================
MALE_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
    "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin",
    "Scott", "Brandon", "Benjamin", "Samuel", "Raymond", "Gregory", "Frank",
    "Alexander", "Patrick", "Jack", "Dennis", "Jerry", "Tyler", "Aaron",
    "Jose", "Adam", "Nathan", "Henry", "Douglas", "Zachary", "Peter", "Kyle",
    "Noah", "Ethan", "Jeremy", "Walter", "Christian", "Keith", "Roger", "Terry",
    "Austin", "Sean", "Gerald", "Carl", "Harold", "Dylan", "Arthur", "Lawrence",
    "Jordan", "Jesse", "Bryan", "Billy", "Bruce", "Gabriel", "Joe", "Logan",
    "Albert", "Willie", "Alan", "Eugene", "Russell", "Vincent", "Philip", "Bobby",
    "Johnny", "Bradley", "Roy", "Ralph", "Randy", "Wayne", "Elijah",
    "Marcus", "Theodore", "Oscar", "Clarence", "Ernest", "Martin", "Craig",
    "Leonard", "Stanley", "Shawn", "Travis", "Russell", "Mason", "Christian",
    "Lawrence", "Bruce", "Dave", "Danny", "Fred", "Bobby", "Jack", "Louis",
    "Phillip", "Sam", "Dean", "Johnathan", "Edgar", "Adrian", "Wesley", "Gordon",
    "Bernard", "Harvey", "Warren", "Ross", "Mitchell", "Simon", "Greg", "Chester",
    "Ben", "Julius", "Curtis", "Glenn", "Antonio", "Randall", "Troy", "Jay",
    "Melvin", "Calvin", "Leroy", "Frankie", "Guy", "Elmer", "Hubert", "Angelo",
    "Clinton", "Chris", "Lewis", "Warren", "Leo", "Mario", "Victor", "Wendell",
    "Clyde", "Cecil", "Marshall", "Milton", "Neil", "Stuart", "Roger", "Max",
    "Luis", "Kent", "Ivan", "Clifford", "Jerry", "Ray", "Lynn", "Gene",
]
FEMALE_NAMES = [
    "Mary", "Patricia", "Linda", "Barbara", "Elizabeth", "Jennifer", "Maria",
    "Susan", "Margaret", "Dorothy", "Lisa", "Nancy", "Karen", "Betty", "Helen",
    "Sandra", "Donna", "Carol", "Ruth", "Sharon", "Michelle", "Laura", "Sarah",
    "Kimberly", "Deborah", "Jessica", "Shirley", "Cynthia", "Angela", "Melissa",
    "Brenda", "Amy", "Anna", "Rebecca", "Virginia", "Kathleen", "Pamela",
    "Martha", "Debra", "Amanda", "Stephanie", "Carolyn", "Christine", "Marie",
    "Janet", "Catherine", "Frances", "Ann", "Joyce", "Diane", "Alice", "Julie",
    "Heather", "Teresa", "Doris", "Gloria", "Evelyn", "Jean", "Cheryl", "Mildred",
    "Katherine", "Joan", "Ashley", "Kimberly", "Michelle", "Dorothy", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
    "Cynthia", "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela",
    "Emma", "Nicole", "Helen", "Samantha", "Katherine", "Christine", "Debra",
    "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather", "Diane",
    "Ruth", "Julie", "Olivia", "Joyce", "Virginia", "Victoria", "Kelly", "Lauren",
    "Christina", "Joan", "Evelyn", "Judith", "Megan", "Andrea", "Cheryl", "Hannah",
    "Jacqueline", "Martha", "Gloria", "Teresa", "Ann", "Sara", "Madison", "Frances",
    "Kathryn", "Janice", "Jean", "Abigail", "Alice", "Judy", "Sophia", "Grace",
    "Danielle", "Brittany", "Mariam", "Molly", "Becky", "Joanne", "Eleanor", "Susan",
    "Gloria", "Carol", "Nancy", "Susan", "Patricia", "Deborah", "Dorothy", "Ashley",
    "Kimberly", "Emily", "Donna", "Michelle", "Sarah", "Amanda", "Stephanie", "Linda",
    "Megan", "Laura", "Jennifer", "Elizabeth", "Nicole", "Melissa", "Heather", "Rachel",
    "Angela", "Amy", "Rebecca", "Katherine", "Samantha", "Diana", "Christina", "Janet",
    "Catherine", "Maria", "Anna", "Brenda", "Pamela", "Emma", "Ruth", "Catherine",
    "Virginia", "Julie", "Joyce", "Diane", "Alice", "Marie", "Frances", "Ann",
    "Evelyn", "Jean", "Cheryl", "Mildred", "Katherine", "Joan", "Doris", "Gloria",
]

def make_gender_dataset(n_per_class, seed=42):
    """Create gender detection dataset from names."""
    rng = np.random.RandomState(seed)
    male = [(n, 0) for n in MALE_NAMES if len(n) >= 3]
    female = [(n, 1) for n in FEMALE_NAMES if len(n) >= 3]
    rng.shuffle(male); rng.shuffle(female)
    male_samp = male[:n_per_class]
    female_samp = female[:n_per_class]
    data = male_samp + female_samp
    rng.shuffle(data)
    return data


# ======================================================================
# TASK 2: Sentiment detection (simple)
# ======================================================================
POSITIVE_SENTS = [
    "I love this, it is wonderful and amazing!", "Great job, well done!",
    "This is fantastic, truly excellent work.", "Brilliant idea, very creative!",
    "Wonderful experience, highly recommend it.", "Perfect solution, exactly right.",
    "Outstanding performance, exceeded expectations.", "Superb quality, very impressive.",
    "Delightful results, beautifully done.", "Excellent outcome, well crafted.",
    "This looks great, very pleasing.", "Fantastic work, thoroughly enjoyed it.",
    "Impressive results, highly skilled.", "Top quality, very professional.",
    "Splendid effort, exceeded all goals.", "Marvelous work, truly remarkable.",
    "Exceptional quality, very satisfying.", "Nice work, looks beautiful.",
    "Admirable performance, very inspiring.", "Magnificent results, first class.",
    "The results are fantastic and exceeded expectations.",
    "This is a brilliant solution that works perfectly.",
    "I am very happy with how this turned out.",
    "The quality is outstanding and impressive.",
    "Excellent work that deserves high praise.",
    "This is a remarkable achievement worth celebrating.",
    "The outcome is wonderful and very satisfying.",
    "I am delighted with the results.",
    "This is truly impressive work.",
    "The performance was excellent and thorough.",
    "A perfect example of great craftsmanship.",
    "Very happy with the outcome.",
    "This exceeded all my expectations.",
    "Absolutely wonderful and excellent.",
    "Brilliant execution and results.",
    "Highly skilled and impressive work.",
    "A truly fantastic achievement.",
    "This looks great and polished.",
    "Superb quality throughout.",
    "Remarkable and outstanding work.",
    "I love how this turned out.",
    "Fantastic results all around.",
    "Very professional and excellent.",
    "Beautiful work, very skilled.",
    "Thoroughly impressed with this.",
    "A masterpiece of quality work.",
    "Exactly what was needed.",
    "This is perfect and excellent.",
    "Great work, very satisfying.",
    "Wonderful and remarkable outcome.",
    "Exceptional in every way.",
    "Very well done and impressive.",
    "A truly excellent result.",
    "Outstanding and superb quality.",
    "Impressive work and great skill.",
    "Admirable and brilliant work.",
    "Splendid and perfect outcome.",
    "Very pleased with this.",
    "Magnificent and excellent.",
    "First-rate quality and work.",
    "This is truly first class.",
    "Remarkable and excellent results.",
    "Highly recommended and impressive.",
    "Wonderful craftsmanship and skill.",
    "Superb and outstanding achievement.",
    "A very impressive outcome.",
    "Fantastic work, very talented.",
    "Excellent and satisfying results.",
    "Beautiful and perfect work.",
    "Brilliantly executed solution.",
    "Very impressive achievement.",
    "Admirable quality throughout.",
    "Great skill and excellent work.",
    "Perfect execution and results.",
    "Outstanding and remarkable work.",
    "This is amazing and wonderful.",
    "Truly excellent and impressive.",
    "Highly skilled and polished.",
    "Splendid work and great results.",
    "Superb quality and excellent.",
    "Impressive and wonderful outcome.",
    "Excellent craftsmanship throughout.",
]
NEGATIVE_SENTS = [
    "This is terrible, awful and disappointing.", "I hate this, it is horrible.",
    "The results are poor and unacceptable.", "Bad work, not good at all.",
    "Disappointing outcome, very dissatisfied.", "Wrong solution, completely off.",
    "Terrible performance, failed completely.", "Low quality work, very poor.",
    "Unpleasant results, badly done.", "Mediocre outcome, not acceptable.",
    "This looks bad and concerning.", "Dreadful work, very unsatisfactory.",
    "Poor results, lacks quality.", "Substandard work, unprofessional.",
    "Lousy effort, missed all goals.", "Dismal work, truly disappointing.",
    "Weak quality, very disappointing.", "Bad work, looks sloppy.",
    "Pathetic performance, not inspiring.", "Shoddy results, second rate.",
    "The results are poor and unacceptable.",
    "This is a terrible solution that failed completely.",
    "I am very unhappy with how this turned out.",
    "The quality is poor and disappointing.",
    "Bad work that deserves criticism.",
    "This is a failed achievement to be ashamed of.",
    "The outcome is awful and very unsatisfying.",
    "I am disappointed with the results.",
    "This is truly disappointing work.",
    "The performance was poor and inadequate.",
    "A terrible example of poor craftsmanship.",
    "Very unhappy with the outcome.",
    "This failed all my expectations.",
    "Absolutely terrible and poor.",
    "Horrible execution and results.",
    "Poorly skilled and disappointing work.",
    "A truly failed achievement.",
    "This looks bad and unfinished.",
    "Poor quality throughout.",
    "Dismal and disappointing work.",
    "I hate how this turned out.",
    "Terrible results all around.",
    "Very unprofessional and poor.",
    "Bad work, very unskilled.",
    "Thoroughly disappointed with this.",
    "A failure of poor quality work.",
    "Not what was needed at all.",
    "This is poor and terrible.",
    "Bad work, very unsatisfying.",
    "Awful and disappointing outcome.",
    "Pathetic in every way.",
    "Very poorly done and unimpressive.",
    "A truly terrible result.",
    "Poor and mediocre quality.",
    "Impressive work and great skill.",
    "Awful and terrible work.",
    "Dreadful and poor outcome.",
    "Very displeased with this.",
    "Dismal and terrible.",
    "Third-rate quality and work.",
    "This is truly third class.",
    "Poor and disappointing results.",
    "Not recommended and unimpressive.",
    "Awful craftsmanship and poor skill.",
    "Bad and mediocre achievement.",
    "A very disappointing outcome.",
    "Terrible work, very incompetent.",
    "Poor and unsatisfactory results.",
    "Bad and imperfect work.",
    "Poorly executed solution.",
    "Very unimpressive achievement.",
    "Mediocre quality throughout.",
    "Poor skill and bad work.",
    "Imperfect execution and results.",
    "Poor and unimpressive work.",
    "This is awful and terrible.",
    "Truly poor and disappointing.",
    "Poorly skilled and unpolished.",
    "Bad work and poor results.",
    "Bad quality and poor work.",
    "Impressive and awful outcome.",
    "Poor craftsmanship throughout.",
]

def make_sentiment_dataset(n_per_class, seed=43):
    """Create sentiment detection dataset."""
    rng = np.random.RandomState(seed)
    pos = [(s, 1) for s in POSITIVE_SENTS]
    neg = [(s, 0) for s in NEGATIVE_SENTS]
    rng.shuffle(pos); rng.shuffle(neg)
    pos_data = (pos * ((n_per_class // len(pos)) + 1))[:n_per_class]
    neg_data = (neg * ((n_per_class // len(neg)) + 1))[:n_per_class]
    data = pos_data + neg_data
    rng.shuffle(data)
    return data


# ======================================================================
# TASK 3: Logical implication detection (causal)
# ======================================================================
CAUSAL_POSITIVE = [
    ("If it rains, the ground gets wet.", "It rained."),
    ("If you heat water, it becomes hot.", "You heated the water."),
    ("If you push a ball, it rolls away.", "You pushed the ball."),
    ("If you add sugar, it tastes sweet.", "You added sugar."),
    ("If you freeze water, it becomes ice.", "You froze the water."),
    ("If you light a match, it produces fire.", "You lit the match."),
    ("If you open the door, the room gets light.", "You opened the door."),
    ("If you cut paper, it separates.", "You cut the paper."),
    ("If you plant a seed, a plant grows.", "You planted the seed."),
    ("If you shine light, it creates brightness.", "You shone light."),
    ("If you stretch a rubber band, it becomes longer.", "You stretched the rubber band."),
    ("If you drop something, it falls down.", "You dropped it."),
    ("If you mix colors, you get a new color.", "You mixed the colors."),
    ("If you unlock a door, it can open.", "You unlocked the door."),
    ("If you plug in a device, it gets power.", "You plugged in the device."),
    ("If it is sunny, the sky is bright.", "It is sunny."),
    ("If you write text, it appears on the page.", "You wrote text."),
    ("If you turn on the light, the room is illuminated.", "You turned on the light."),
    ("If you close the window, wind stops coming in.", "You closed the window."),
    ("If you press a button, the machine activates.", "You pressed the button."),
    ("If you pour water, the glass fills.", "You poured water."),
    ("If you mix oil and water, they separate.", "You mixed oil and water."),
    ("If you heat ice, it melts.", "You heated the ice."),
    ("If you shake a bottle, the liquid bubbles.", "You shook the bottle."),
    ("If you cover a fire, it stops burning.", "You covered the fire."),
    ("If you salt food, it tastes salty.", "You salted the food."),
    ("If you inflate a balloon, it expands.", "You inflated the balloon."),
    ("If you bend a stick, it curves.", "You bent the stick."),
    ("If you color on paper, marks appear.", "You colored on paper."),
    ("If you turn a key, the lock opens.", "You turned the key."),
    ("If you open a tap, water flows out.", "You opened the tap."),
    ("If you touch fire, it burns.", "You touched fire."),
    ("If you press clay, it changes shape.", "You pressed the clay."),
    ("If you wind a clock, it starts ticking.", "You wound the clock."),
    ("If you scratch a matchbox, fire appears.", "You scratched the matchbox."),
    ("If you charge a battery, it stores energy.", "You charged the battery."),
    ("If you point a telescope, you see distant objects.", "You pointed the telescope."),
    ("If you stir liquid, it rotates.", "You stirred the liquid."),
    ("If you shade an object, it gets dark.", "You shaded the object."),
    ("If you wrap a gift, it becomes presentable.", "You wrapped the gift."),
    ("If you sharpen a pencil, it becomes pointed.", "You sharpened the pencil."),
]
CAUSAL_NEGATIVE = [
    ("If it rains, the ground gets wet.", "The ground is dry."),
    ("If you heat water, it becomes hot.", "The water is cold."),
    ("If you push a ball, it rolls away.", "The ball stays still."),
    ("If you add sugar, it tastes sweet.", "The food tastes bitter."),
    ("If you freeze water, it becomes ice.", "The water remains liquid."),
    ("If you light a match, it produces fire.", "No fire appeared."),
    ("If you open the door, the room gets light.", "The room stays dark."),
    ("If you cut paper, it separates.", "The paper stays intact."),
    ("If you plant a seed, a plant grows.", "Nothing grew."),
    ("If you shine light, it creates brightness.", "The area stays dark."),
    ("If you stretch a rubber band, it becomes longer.", "The band stays the same size."),
    ("If you drop something, it falls down.", "The object floated."),
    ("If you mix colors, you get a new color.", "No color change happened."),
    ("If you unlock a door, it can open.", "The door is stuck."),
    ("If you plug in a device, it gets power.", "The device is off."),
    ("If it is sunny, the sky is bright.", "The sky is overcast."),
    ("If you write text, it appears on the page.", "Nothing was written."),
    ("If you turn on the light, the room is illuminated.", "The room is dark."),
    ("If you close the window, wind stops coming in.", "Wind is still coming in."),
    ("If you press a button, the machine activates.", "Nothing happened."),
    ("If you pour water, the glass fills.", "The glass stayed empty."),
    ("If you mix oil and water, they mix completely.", "They remain separated."),
    ("If you heat ice, it melts.", "The ice got harder."),
    ("If you shake a bottle, the liquid settles.", "Bubbles formed."),
    ("If you cover a fire, it continues burning.", "The fire went out."),
    ("If you salt food, it tastes salty.", "The food is sweet."),
    ("If you inflate a balloon, it expands.", "The balloon shrank."),
    ("If you bend a stick, it curves.", "The stick stayed straight."),
    ("If you color on paper, marks appear.", "The paper stayed blank."),
    ("If you turn a key, the lock opens.", "The lock did not open."),
    ("If you open a tap, water flows out.", "No water came out."),
    ("If you touch fire, it burns.", "Nothing happened."),
    ("If you press clay, it changes shape.", "The clay was already shaped."),
    ("If you wind a clock, it starts ticking.", "The clock is silent."),
    ("If you scratch a matchbox, fire appears.", "No spark appeared."),
    ("If you charge a battery, it stores energy.", "The battery is dead."),
    ("If you point a telescope, you see distant objects.", "Everything is blurry."),
    ("If you stir liquid, it rotates.", "The liquid is still."),
    ("If you shade an object, it gets dark.", "The object is bright."),
    ("If you wrap a gift, it becomes presentable.", "The gift looks messy."),
    ("If you sharpen a pencil, it becomes pointed.", "The pencil is dull."),
    ("If you cool water, it freezes.", "The water stayed warm."),
    ("If you clap your hands, it makes a sound.", "No sound was heard."),
]

def make_causal_dataset(n_per_class, seed=44):
    """Create causal reasoning dataset from premise + context pairs."""
    rng = np.random.RandomState(seed)
    pos = [(f"{p}\nContext: {c}", 1) for p, c in CAUSAL_POSITIVE]
    neg = [(f"{p}\nContext: {c}", 0) for p, c in CAUSAL_NEGATIVE]
    rng.shuffle(pos); rng.shuffle(neg)
    pos_data = (pos * ((n_per_class // len(pos)) + 2))[:n_per_class]
    neg_data = (neg * ((n_per_class // len(neg)) + 2))[:n_per_class]
    data = pos_data + neg_data
    rng.shuffle(data)
    return data


# ======================================================================
# Feature evaluation: batch-based evaluation
# ======================================================================
def run_all_features_batch(model, SAE_LAYER, sae, feature_indices, texts_list, labels):
    """Batch evaluation: run model + SAE for all texts, compute AUCs per feature."""
    from sklearn.metrics import roc_auc_score

    hook_name = f"blocks.{SAE_LAYER}.hook_resid_pre"
    all_acts = []

    for i in range(0, len(texts_list), BATCH_SIZE):
        batch_texts = texts_list[i:i+BATCH_SIZE]
        tokens = model.tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).input_ids.to(model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
                remove_batch_dim=False,
            )
        acts = cache[hook_name]
        sae_acts = sae.encode(acts).detach()
        feat_acts = sae_acts[:, :, feature_indices].mean(dim=1).cpu().numpy()
        all_acts.append(feat_acts)
        del cache, acts, sae_acts
        torch.cuda.empty_cache()

    all_acts = np.concatenate(all_acts, axis=0)
    labels_arr = np.array(labels)

    results = {}
    for j, feat_idx in enumerate(feature_indices):
        try:
            auc = roc_auc_score(labels_arr, -all_acts[:, j])
        except Exception:
            auc = 0.5
        results[feat_idx] = auc

    del all_acts
    return results


def evaluate_model_features(model, SAE_LAYER, sae, feature_bins, absorption_scores, model_name):
    """Evaluate all features on all 3 tasks for a model."""
    log(f"\n=== Evaluating {model_name} features ===")

    all_features = feature_bins["low"] + feature_bins["mid"] + feature_bins["high"]
    log(f"  Features: {len(all_features)} total")

    # Build datasets
    gender_ds = make_gender_dataset(N_SAMPLES_PER_CLASS, SEED)
    sentiment_ds = make_sentiment_dataset(N_SAMPLES_PER_CLASS, SEED + 1)
    causal_ds = make_causal_dataset(N_SAMPLES_PER_CLASS, SEED + 2)

    gender_texts = [item[0] for item in gender_ds]
    gender_labels = [item[1] for item in gender_ds]
    sentiment_texts = [item[0] for item in sentiment_ds]
    sentiment_labels = [item[1] for item in sentiment_ds]
    causal_texts = [item[0] for item in causal_ds]
    causal_labels = [item[1] for item in causal_ds]

    log(f"  Task 1 (gender): {len(gender_texts)} samples")
    log(f"  Task 2 (sentiment): {len(sentiment_texts)} samples")
    log(f"  Task 3 (causal): {len(causal_texts)} samples")

    log(f"  Evaluating on Task 1 (gender)...")
    gender_aucs = run_all_features_batch(model, SAE_LAYER, sae, all_features, gender_texts, gender_labels)

    log(f"  Evaluating on Task 2 (sentiment)...")
    sentiment_aucs = run_all_features_batch(model, SAE_LAYER, sae, all_features, sentiment_texts, sentiment_labels)

    log(f"  Evaluating on Task 3 (causal)...")
    causal_aucs = run_all_features_batch(model, SAE_LAYER, sae, all_features, causal_texts, causal_labels)

    # Aggregate by bin
    results = {}
    for bin_name in ["low", "mid", "high"]:
        feats = feature_bins[bin_name]
        g_aucs = [gender_aucs[f] for f in feats]
        s_aucs = [sentiment_aucs[f] for f in feats]
        c_aucs = [causal_aucs[f] for f in feats]

        results[bin_name] = {
            "n_features": len(feats),
            "mean_uas": float(np.mean([absorption_scores[f] for f in feats])),
            "gender_auc_mean": float(np.mean(g_aucs)),
            "gender_auc_std": float(np.std(g_aucs)),
            "gender_aucs": {str(f): float(gender_aucs[f]) for f in feats},
            "sentiment_auc_mean": float(np.mean(s_aucs)),
            "sentiment_auc_std": float(np.std(s_aucs)),
            "sentiment_aucs": {str(f): float(sentiment_aucs[f]) for f in feats},
            "causal_auc_mean": float(np.mean(c_aucs)),
            "causal_auc_std": float(np.std(c_aucs)),
            "causal_aucs": {str(f): float(causal_aucs[f]) for f in feats},
            "feature_indices": feats,
        }

    return results, {
        "gender": gender_aucs,
        "sentiment": sentiment_aucs,
        "causal": causal_aucs,
    }


def collect_activations_simple(model, sae, hook_name, n_batches, batch_size=64):
    """Collect SAE activations on pile data. Uses stop_at_layer to avoid caching."""
    from datasets import load_dataset
    log(f"Collecting activations ({n_batches} batches x {batch_size} samples)...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    dataset = dataset.shuffle(seed=SEED)

    activations_list = []
    actual_layer = int(hook_name.split(".")[1])

    for i, batch in enumerate(dataset.iter(batch_size=batch_size)):
        if i >= n_batches:
            break
        tokens = model.tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=128
        ).input_ids.to(model.cfg.device)

        # Use stop_at_layer to limit computation and avoid caching all layers
        with torch.no_grad():
            # Run only up to and including the target layer
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
                remove_batch_dim=False,
                stop_at_layer=actual_layer + 1,
            )
        acts = cache[hook_name]
        sae_acts = sae.encode(acts).detach()
        activations_list.append(sae_acts.cpu())
        del cache, acts, sae_acts
        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            log(f"  Batch {i+1}/{n_batches}")

    activations = torch.cat(activations_list, dim=1)
    del activations_list
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  Collected {activations.shape[1]} tokens of activations")
    return activations


def main():
    set_seed(SEED)
    log(f"[{datetime.now().isoformat()}] Starting full_h5: downstream task analysis")
    log(f"  Device: {DEVICE}")
    log(f"  Features per model: {N_FEATURES_PER_MODEL}")
    log(f"  Samples per class per task: {N_SAMPLES_PER_CLASS}")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # =============================================================
    # PART 1: GPT-2 Small Layer 8
    # =============================================================
    log(f"\n{'='*60}")
    log("PART 1: GPT-2 Small Layer 8")
    log('='*60)

    log("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.eval()
    model.to(DEVICE)

    log("Loading SAE for GPT-2 L8...")
    sae_gpt2 = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=DEVICE,
    )
    sae_gpt2.eval()
    log(f"  SAE d_sae={sae_gpt2.cfg.d_sae}, n_features={sae_gpt2.W_dec.shape[0]}")

    # Collect activations
    activations_gpt2 = collect_activations_simple(model, sae_gpt2, "blocks.8.hook_resid_pre", n_batches=20, batch_size=64)

    top_features_gpt2 = get_top_activated_features(activations_gpt2, n_features_per_bin * 3)
    log(f"  Top {len(top_features_gpt2)} features selected")

    absorption_gpt2 = compute_neg_cos_sim_var(sae_gpt2, top_features_gpt2)
    feature_bins_gpt2 = bin_features_by_absorption(absorption_gpt2, n_features_per_bin, SEED)
    log(f"  GPT-2 bins: low={len(feature_bins_gpt2['low'])}, mid={len(feature_bins_gpt2['mid'])}, high={len(feature_bins_gpt2['high'])}")

    gpt2_results, gpt2_feature_aucs = evaluate_model_features(
        model, 8, sae_gpt2, feature_bins_gpt2, absorption_gpt2, "GPT-2 Small L8"
    )

    del model, sae_gpt2, activations_gpt2
    gc.collect()
    torch.cuda.empty_cache()

    # =============================================================
    # PART 2: GPT-2 Small Layer 6 (Gemma-2B not available, using L6 as second layer)
    # =============================================================
    log(f"\n{'='*60}")
    log("PART 2: GPT-2 Small Layer 6")
    log('='*60)

    LAYER_2 = 6
    log(f"Loading GPT-2 Small Layer {LAYER_2}...")
    model2 = HookedTransformer.from_pretrained("gpt2-small")
    model2.eval()
    model2.to(DEVICE)

    log(f"Loading SAE for GPT-2 L{LAYER_2}...")
    sae_l6 = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER_2}.hook_resid_pre",
        device=DEVICE,
    )
    sae_l6.eval()
    log(f"  SAE d_sae={sae_l6.cfg.d_sae}, n_features={sae_l6.W_dec.shape[0]}")

    hook_name_l6 = f"blocks.{LAYER_2}.hook_resid_pre"
    activations_l6 = collect_activations_simple(model2, sae_l6, hook_name_l6, n_batches=20, batch_size=64)

    top_features_l6 = get_top_activated_features(activations_l6, n_features_per_bin * 3)
    log(f"  Top {len(top_features_l6)} features selected")

    absorption_l6 = compute_neg_cos_sim_var(sae_l6, top_features_l6)
    feature_bins_l6 = bin_features_by_absorption(absorption_l6, n_features_per_bin, SEED + 100)
    log(f"  GPT-2 L{LAYER_2} bins: low={len(feature_bins_l6['low'])}, mid={len(feature_bins_l6['mid'])}, high={len(feature_bins_l6['high'])}")

    l6_results, l6_feature_aucs = evaluate_model_features(
        model2, LAYER_2, sae_l6, feature_bins_l6, absorption_l6, f"GPT-2 Small L{LAYER_2}"
    )

    del model2, sae_l6, activations_l6
    gc.collect()
    torch.cuda.empty_cache()

    # =============================================================
    # PART 3: Aggregate results
    # =============================================================
    log(f"\n{'='*60}")
    log("Aggregating results...")
    log('='*60)

    combined_results = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_features_per_model": N_FEATURES_PER_MODEL,
            "n_samples_per_class": N_SAMPLES_PER_CLASS,
            "seed": SEED,
            "device": DEVICE,
            "absorption_metric": "neg_cos_sim_var",
        },
        "model_results": {
            "gpt2_small_l8": {
                "results_by_bin": gpt2_results,
                "absorption_scores": {str(k): float(v) for k, v in absorption_gpt2.items()},
                "feature_aucs": {
                    "gender": {str(k): float(v) for k, v in gpt2_feature_aucs["gender"].items()},
                    "sentiment": {str(k): float(v) for k, v in gpt2_feature_aucs["sentiment"].items()},
                    "causal": {str(k): float(v) for k, v in gpt2_feature_aucs["causal"].items()},
                },
            },
            "gpt2_small_l6": {
                "results_by_bin": l6_results,
                "absorption_scores": {str(k): float(v) for k, v in absorption_l6.items()},
                "feature_aucs": {
                    "gender": {str(k): float(v) for k, v in l6_feature_aucs["gender"].items()},
                    "sentiment": {str(k): float(v) for k, v in l6_feature_aucs["sentiment"].items()},
                    "causal": {str(k): float(v) for k, v in l6_feature_aucs["causal"].items()},
                },
            },
        },
    }

    def compute_task_delta(results_by_bin, task_key):
        low_auc = results_by_bin["low"][f"{task_key}_auc_mean"]
        mid_auc = results_by_bin["mid"][f"{task_key}_auc_mean"]
        high_auc = results_by_bin["high"][f"{task_key}_auc_mean"]
        return abs(high_auc - low_auc)

    log("\n--- GPT-2 L8 Results ---")
    gpt2_pass_simple = True
    gpt2_pass_causal = False
    for task in ["gender", "sentiment"]:
        delta = compute_task_delta(gpt2_results, task)
        log(f"  {task}: delta={delta:.3f}")
        if delta > 0.03:
            gpt2_pass_simple = False

    causal_delta = compute_task_delta(gpt2_results, "causal")
    log(f"  causal: delta={causal_delta:.3f}")
    if causal_delta > 0.08:
        gpt2_pass_causal = True

    log("\n--- GPT-2 L6 Results ---")
    l6_pass_simple = True
    l6_pass_causal = False
    for task in ["gender", "sentiment"]:
        delta = compute_task_delta(l6_results, task)
        log(f"  {task}: delta={delta:.3f}")
        if delta > 0.03:
            l6_pass_simple = False

    causal_delta_l6 = compute_task_delta(l6_results, "causal")
    log(f"  causal: delta={causal_delta_l6:.3f}")
    if causal_delta_l6 > 0.08:
        l6_pass_causal = True

    h5_confirmed = (
        (gpt2_pass_simple or l6_pass_simple) and
        (gpt2_pass_causal or l6_pass_causal)
    )

    combined_results["pass_criteria"] = {
        "gpt2_simple_delta_lte_3pct": gpt2_pass_simple,
        "gpt2_causal_delta_gt_8pct": gpt2_pass_causal,
        "l6_simple_delta_lte_3pct": l6_pass_simple,
        "l6_causal_delta_gt_8pct": l6_pass_causal,
        "h5_confirmed": h5_confirmed,
    }

    # Summary
    for model_key, model_results in [("gpt2_l8", gpt2_results), ("gpt2_l6", l6_results)]:
        for task in ["gender", "sentiment", "causal"]:
            vals = [model_results[b][f"{task}_auc_mean"] for b in ["low", "mid", "high"]]
            log(f"  {model_key} {task}: low={vals[0]:.3f}, mid={vals[1]:.3f}, high={vals[2]:.3f}, delta={abs(vals[2]-vals[0]):.3f}")

    combined_results["summary"] = {
        "total_features": N_FEATURES_PER_MODEL * 2,
        "h5_interpretation": (
            "H5 CONFIRMED: absorption tolerable for simple tasks, harmful for causal reasoning"
            if h5_confirmed else
            "H5 NOT CONFIRMED: see individual task delta results for details"
        ),
    }

    # Save results
    out_path = RESULTS_DIR / f"{TASK_ID}_downstream_tasks.json"
    with open(out_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    log(f"\nResults saved to {out_path}")

    done_path = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_path.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "h5_confirmed": h5_confirmed,
    }))
    log(f"\n[{datetime.now().isoformat()}] full_h5 complete!")
    log(f"  H5 confirmed: {h5_confirmed}")


if __name__ == "__main__":
    main()
