#!/usr/bin/env python3
"""
Pilot Experiment 1: Generate synthetic 3-level feature hierarchies
for testing SAE absorption under random baseline scrutiny.

Creates controlled parent-child-grandchild relationships where parent
direction = convex combination of children (simulating absorption).
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
PILOT_SAMPLES = 100
FULL_SAMPLES = 10000
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

def create_feature_hierarchy(d_model=128, hierarchy_type="animals"):
    """
    Create a 3-level feature hierarchy with controlled relationships.

    Level 0 (parent): "animals"
    Level 1 (children): "dogs", "cats"
    Level 2 (grandchildren): "poodles", "huskies", "tabbies", "persians"
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Generate orthogonal base directions for grandchildren (Level 2)
    grandchild_count = 4
    grandchild_directions = []
    for i in range(grandchild_count):
        direction = np.random.randn(d_model)
        direction = direction / np.linalg.norm(direction)
        grandchild_directions.append(direction)

    # Generate children (Level 1) as combinations of grandchildren
    # dogs = poodles + huskies, cats = tabbies + persians
    dogs_direction = 0.5 * grandchild_directions[0] + 0.5 * grandchild_directions[1]
    cats_direction = 0.5 * grandchild_directions[2] + 0.5 * grandchild_directions[3]

    # Normalize
    dogs_direction = dogs_direction / np.linalg.norm(dogs_direction)
    cats_direction = cats_direction / np.linalg.norm(cats_direction)

    # Generate parent (Level 0) as combination of children
    # animals = dogs + cats (absorption scenario: parent = linear combo of children)
    animals_direction = 0.5 * dogs_direction + 0.5 * cats_direction
    animals_direction = animals_direction / np.linalg.norm(animals_direction)

    return {
        "hierarchy_type": hierarchy_type,
        "d_model": d_model,
        "levels": {
            "parent": {
                "name": "animals",
                "direction": animals_direction.tolist(),
                "level": 0,
                "children": ["dogs", "cats"]
            },
            "children": [
                {
                    "name": "dogs",
                    "direction": dogs_direction.tolist(),
                    "level": 1,
                    "children": ["poodles", "huskies"]
                },
                {
                    "name": "cats",
                    "direction": cats_direction.tolist(),
                    "level": 1,
                    "children": ["tabbies", "persians"]
                }
            ],
            "grandchildren": [
                {
                    "name": "poodles",
                    "direction": grandchild_directions[0].tolist(),
                    "level": 2
                },
                {
                    "name": "huskies",
                    "direction": grandchild_directions[1].tolist(),
                    "level": 2
                },
                {
                    "name": "tabbies",
                    "direction": grandchild_directions[2].tolist(),
                    "level": 2
                },
                {
                    "name": "persians",
                    "direction": grandchild_directions[3].tolist(),
                    "level": 2
                }
            ]
        }
    }

def generate_synthetic_activations(hierarchy, n_samples, d_model=128, seed=42):
    """
    Generate synthetic MLP activations based on the feature hierarchy.

    Creates activations where:
    - Features fire based on semantic categories
    - Parent features fire when children fire (simulating absorption)
    - Adds noise for realism
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract directions
    grandchild_dirs = [np.array(hierarchy["levels"]["grandchildren"][i]["direction"]) for i in range(4)]
    dogs_dir = np.array(hierarchy["levels"]["children"][0]["direction"])
    cats_dir = np.array(hierarchy["levels"]["children"][1]["direction"])
    animals_dir = np.array(hierarchy["levels"]["parent"]["direction"])

    # Create orthogonal basis including the 4 hierarchy directions
    all_dirs = [animals_dir, dogs_dir, cats_dir] + grandchild_dirs

    activations = []
    feature_labels = []

    for i in range(n_samples):
        # Sample a category
        category = np.random.choice(["animals", "dogs", "cats", "poodles", "huskies", "tabbies", "persians", "other"])

        # Create activation vector
        x = np.random.randn(d_model) * 0.1  # Background noise

        if category == "poodles":
            x = x + 5.0 * grandchild_dirs[0]
        elif category == "huskies":
            x = x + 5.0 * grandchild_dirs[1]
        elif category == "tabbies":
            x = x + 5.0 * grandchild_dirs[2]
        elif category == "persians":
            x = x + 5.0 * grandchild_dirs[3]
        elif category == "dogs":
            x = x + 5.0 * dogs_dir
        elif category == "cats":
            x = x + 5.0 * cats_dir
        elif category == "animals":
            # Animals fires when both dogs and cats fire (absorption scenario)
            x = x + 5.0 * animals_dir  # Parent fires
            x = x + 3.0 * dogs_dir  # Child also fires
            x = x + 3.0 * cats_dir  # Child also fires

        activations.append(x)
        feature_labels.append(category)

    return np.array(activations), feature_labels

def validate_hierarchy(activations, hierarchy, sample_size=20):
    """
    Validate that the hierarchy structure is detectable via cosine similarity.
    """
    from scipy.spatial.distance import cosine

    d_model = hierarchy["d_model"]
    grandchild_dirs = [np.array(hierarchy["levels"]["grandchildren"][i]["direction"]) for i in range(4)]
    dogs_dir = np.array(hierarchy["levels"]["children"][0]["direction"])
    cats_dir = np.array(hierarchy["levels"]["children"][1]["direction"])
    animals_dir = np.array(hierarchy["levels"]["parent"]["direction"])

    # Test 1: Grandchildren should be orthogonal to each other
    grandchild_orthogonality = []
    for i in range(4):
        for j in range(i+1, 4):
            sim = 1 - cosine(grandchild_dirs[i], grandchild_dirs[j])
            grandchild_orthogonality.append(sim)

    # Test 2: Children should be similar to their grandchildren
    dogs_sim = 1 - cosine(dogs_dir, 0.5*grandchild_dirs[0] + 0.5*grandchild_dirs[1])
    cats_sim = 1 - cosine(cats_dir, 0.5*grandchild_dirs[2] + 0.5*grandchild_dirs[3])

    # Test 3: Parent should be similar to children (absorption scenario)
    parent_dogs_sim = 1 - cosine(animals_dir, dogs_dir)
    parent_cats_sim = 1 - cosine(animals_dir, cats_dir)

    validation = {
        "grandchild_orthogonality_mean": float(np.mean(grandchild_orthogonality)),
        "dogs_grandchild_similarity": float(dogs_sim),
        "cats_grandchild_similarity": float(cats_sim),
        "parent_children_similarity": float((parent_dogs_sim + parent_cats_sim) / 2),
        "hierarchy_detectable": bool(
            np.mean(grandchild_orthogonality) < 0.3 and  # Grandchildren orthogonal
            dogs_sim > 0.8 and cats_sim > 0.8 and  # Children match grandchildren
            parent_dogs_sim > 0.7 and parent_cats_sim > 0.7  # Parent matches children
        )
    }

    return validation

def main():
    print("=" * 60)
    print("Pilot Experiment 1: Synthetic Feature Hierarchy Generation")
    print("=" * 60)

    # Create hierarchy
    print("\n[1/4] Creating 3-level feature hierarchy...")
    hierarchy = create_feature_hierarchy(d_model=128)

    # Generate pilot activations
    print(f"[2/4] Generating {PILOT_SAMPLES} pilot samples...")
    activations_pilot, labels_pilot = generate_synthetic_activations(
        hierarchy, PILOT_SAMPLES, d_model=128, seed=SEED
    )

    # Validate hierarchy structure
    print("[3/4] Validating hierarchy structure...")
    validation = validate_hierarchy(activations_pilot, hierarchy)

    # Generate full dataset (for later use)
    print(f"[4/4] Generating {FULL_SAMPLES} full samples...")
    activations_full, labels_full = generate_synthetic_activations(
        hierarchy, FULL_SAMPLES, d_model=128, seed=SEED
    )

    # Save results
    output_dir = WORKSPACE / "data"
    output_dir.mkdir(exist_ok=True)

    # Hierarchy definition
    hierarchy_path = output_dir / "synthetic_hierarchy.json"
    with open(hierarchy_path, "w") as f:
        json.dump(hierarchy, f, indent=2)
    print(f"  Saved: {hierarchy_path}")

    # Pilot dataset
    pilot_data = {
        "activations": activations_pilot.tolist(),
        "labels": labels_pilot,
        "d_model": 128,
        "n_samples": PILOT_SAMPLES,
        "seed": SEED,
        "created_at": datetime.now().isoformat()
    }
    pilot_path = output_dir / "pilot_activations.json"
    with open(pilot_path, "w") as f:
        json.dump(pilot_data, f)
    print(f"  Saved: {pilot_path}")

    # Full dataset
    full_data = {
        "activations": activations_full.tolist(),
        "labels": labels_full,
        "d_model": 128,
        "n_samples": FULL_SAMPLES,
        "seed": SEED,
        "created_at": datetime.now().isoformat()
    }
    full_path = output_dir / "full_activations.json"
    with open(full_path, "w") as f:
        json.dump(full_data, f)
    print(f"  Saved: {full_path}")

    # Validation report
    validation_path = output_dir / "hierarchy_validation.json"
    with open(validation_path, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"  Saved: {validation_path}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Grandchild orthogonality: {validation['grandchild_orthogonality_mean']:.4f} (expected < 0.3)")
    print(f"  Dogs-grandchildren similarity: {validation['dogs_grandchild_similarity']:.4f} (expected > 0.8)")
    print(f"  Cats-grandchildren similarity: {validation['cats_grandchild_similarity']:.4f} (expected > 0.8)")
    print(f"  Parent-children similarity: {validation['parent_children_similarity']:.4f} (expected > 0.7)")
    print(f"  Hierarchy detectable: {validation['hierarchy_detectable']}")

    pass_criteria = validation['hierarchy_detectable']
    print(f"\n  PILOT PASS: {pass_criteria}")

    if pass_criteria:
        print("\n[SUCCESS] Hierarchy structure validated. Proceed to Task 2 (SAE training).")
    else:
        print("\n[WARNING] Hierarchy validation failed. Check structure.")

    return {
        "status": "success" if pass_criteria else "partial",
        "hierarchy_path": str(hierarchy_path),
        "pilot_path": str(pilot_path),
        "validation": validation
    }

if __name__ == "__main__":
    result = main()
    print(f"\nFinal result: {result}")
