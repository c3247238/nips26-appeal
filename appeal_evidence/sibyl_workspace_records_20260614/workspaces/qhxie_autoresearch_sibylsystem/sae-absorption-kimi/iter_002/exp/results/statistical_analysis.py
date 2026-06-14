#!/usr/bin/env python3
"""Statistical analysis and figure generation for SAE absorption construct validity study."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')

RESULTS_DIR = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/results/full')
OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_pearsonr(x, y, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(x)
    r_obs, _ = stats.pearsonr(x, y)
    r_boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        try:
            with np.errstate(invalid='ignore'):
                r, _ = stats.pearsonr(x[idx], y[idx])
            if not np.isfinite(r):
                continue
            r_boot.append(r)
        except Exception:
            continue
    r_boot = np.array(r_boot)
    if len(r_boot) == 0:
        return r_obs, float('nan'), float('nan'), r_boot
    ci_lower = float(np.percentile(r_boot, 2.5))
    ci_upper = float(np.percentile(r_boot, 97.5))
    return r_obs, ci_lower, ci_upper, r_boot


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    # Load all result files
    firstletter = load_json(RESULTS_DIR / 'firstletter_pythia_results.json')
    semantic = load_json(RESULTS_DIR / 'semantic_hierarchy_pythia_results.json')
    nonhierarchy = load_json(RESULTS_DIR / 'nonhierarchy_control_pythia_results.json')
    tau_fs = load_json(RESULTS_DIR / 'tau_fs_robustness_results.json')
    gpt2 = load_json(RESULTS_DIR / 'gpt2_replication_results.json')

    # Build lookup by family
    fl_lookup = {c['family']: c for c in firstletter['checkpoints']}
    sem_lookup = {r['family']: r for r in semantic['sae_results']}
    nh_lookup = {r['family']: r for r in nonhierarchy['sae_results']}

    families = ['BatchTopK', 'GatedSAE', 'JumpRelu', 'MatryoshkaBatchTopK',
                'PAnneal', 'Standard', 'TopK', 'Random']

    # Extract vectors
    fl_scores = np.array([fl_lookup[f]['official_absorption_full'] for f in families])
    sem_scores = np.array([sem_lookup[f]['mean_hierarchy_absorption'] for f in families])
    nh_scores = np.array([nh_lookup[f]['mean_pair_absorption'] for f in families])

    # H1: Pearson r first-letter vs semantic (excluding Random for main test)
    main_mask = np.array([f != 'Random' for f in families])
    r_h1, ci_l_h1, ci_u_h1, _ = bootstrap_pearsonr(fl_scores[main_mask], sem_scores[main_mask])

    # With Random included
    r_h1_all, ci_l_h1_all, ci_u_h1_all, _ = bootstrap_pearsonr(fl_scores, sem_scores)

    # H2: paired t-test semantic vs non-hierarchy (excluding Random)
    t_h2, p_h2 = stats.ttest_rel(sem_scores[main_mask], nh_scores[main_mask])

    # H1 alternative: first-letter vs non-hierarchy (excluding Random)
    r_h1_nh, ci_l_h1_nh, ci_u_h1_nh, _ = bootstrap_pearsonr(fl_scores[main_mask], nh_scores[main_mask])

    # Per-hierarchy probe AUROC table
    hierarchy_names = list(semantic['hierarchy_probe_aurocs'].keys())
    per_hierarchy_auroc = {h: semantic['hierarchy_probe_aurocs'][h] for h in hierarchy_names}
    per_hierarchy_absorption = {}
    for h in hierarchy_names:
        vals = []
        for f in families:
            for hh in sem_lookup[f]['hierarchies']:
                if hh['parent'] == h:
                    vals.append(hh['absorption'])
                    break
        per_hierarchy_absorption[h] = vals

    # Tau_fs robustness (reuse from tau_fs file but recompute cleanly)
    tau_robustness = []
    for entry in tau_fs['correlation_analysis']:
        tau_robustness.append({
            'tau_fs': entry['tau_fs'],
            'pearson_r_firstletter_semantic': entry['pearson_r_firstletter_semantic'],
            'bootstrap_ci_lower': entry['bootstrap_ci_lower_firstletter_semantic'],
            'bootstrap_ci_upper': entry['bootstrap_ci_upper_firstletter_semantic'],
            'paired_ttest_semantic_vs_nonhierarchy_t': entry['paired_ttest_semantic_vs_nonhierarchy_t'],
            'paired_ttest_semantic_vs_nonhierarchy_p': entry['paired_ttest_semantic_vs_nonhierarchy_p'],
        })

    # GPT-2 replication summary
    gpt2_summary = []
    for r in gpt2['sae_results']:
        gpt2_summary.append({
            'family': r['family'],
            'mean_hierarchy_absorption': r['mean_hierarchy_absorption'],
            'mean_pair_absorption': r['mean_pair_absorption'],
        })

    # ======================= FIGURES =======================

    # Figure 1: Architecture ranking comparison (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(families))
    width = 0.35
    bars1 = ax.bar(x - width/2, fl_scores, width, label='First-Letter Absorption', color='steelblue')
    bars2 = ax.bar(x + width/2, sem_scores, width, label='Semantic-Hierarchy Absorption', color='coral')
    ax.set_ylabel('Absorption Score')
    ax.set_title('Architecture Ranking: First-Letter vs Semantic-Hierarchy Absorption')
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(max(fl_scores), max(sem_scores)) * 1.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig1_architecture_ranking.png', dpi=300)
    plt.close(fig)

    # Figure 2: Scatter plot first-letter vs semantic-hierarchy
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue' if f != 'Random' else 'gray' for f in families]
    ax.scatter(fl_scores, sem_scores, c=colors, s=120, edgecolors='black', zorder=3)
    for i, f in enumerate(families):
        ax.annotate(f, (fl_scores[i], sem_scores[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    # Regression line (excluding Random)
    z = np.polyfit(fl_scores[main_mask], sem_scores[main_mask], 1)
    p = np.poly1d(z)
    x_line = np.linspace(fl_scores[main_mask].min(), fl_scores[main_mask].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Regression (excl. Random)')
    ax.set_xlabel('First-Letter Absorption Score')
    ax.set_ylabel('Semantic-Hierarchy Absorption Score')
    ax.set_title(f'Construct Validity Test: Pearson r = {r_h1:.3f} [{ci_l_h1:.3f}, {ci_u_h1:.3f}]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig2_firstletter_vs_semantic_scatter.png', dpi=300)
    plt.close(fig)

    # Figure 3: Bar chart semantic vs non-hierarchy per architecture
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, sem_scores, width, label='Semantic-Hierarchy', color='coral')
    bars2 = ax.bar(x + width/2, nh_scores, width, label='Non-Hierarchy Control', color='seagreen')
    ax.set_ylabel('Absorption Score')
    ax.set_title('Hierarchy Specificity: Semantic vs Non-Hierarchy Control')
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(max(sem_scores), max(nh_scores)) * 1.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig3_semantic_vs_nonhierarchy.png', dpi=300)
    plt.close(fig)

    # Figure 4: Tau_fs robustness
    fig, ax = plt.subplots(figsize=(8, 5))
    taus = [t['tau_fs'] for t in tau_robustness]
    rs = [t['pearson_r_firstletter_semantic'] for t in tau_robustness]
    ci_lows = [t['bootstrap_ci_lower'] for t in tau_robustness]
    ci_ups = [t['bootstrap_ci_upper'] for t in tau_robustness]
    ax.errorbar(taus, rs, yerr=[np.array(rs) - np.array(ci_lows), np.array(ci_ups) - np.array(rs)],
                fmt='o-', capsize=5, color='darkviolet', linewidth=2, markersize=8)
    ax.axhline(0.6, color='green', linestyle='--', alpha=0.7, label='H1 threshold (r = 0.6)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='r = 0')
    ax.set_xlabel('tau_fs')
    ax.set_ylabel('Pearson r (First-Letter vs Semantic)')
    ax.set_title('Robustness Analysis: Correlation Across tau_fs Values')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_tau_fs_robustness.png', dpi=300)
    plt.close(fig)

    # Figure 5: GPT-2 replication comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    gpt2_fams = [g['family'] for g in gpt2_summary]
    gpt2_sem = [g['mean_hierarchy_absorption'] for g in gpt2_summary]
    gpt2_nh = [g['mean_pair_absorption'] for g in gpt2_summary]
    x2 = np.arange(len(gpt2_fams))
    ax.bar(x2 - width/2, gpt2_sem, width, label='Semantic-Hierarchy', color='coral')
    ax.bar(x2 + width/2, gpt2_nh, width, label='Non-Hierarchy Control', color='seagreen')
    ax.set_ylabel('Absorption Score')
    ax.set_title('GPT-2 Replication: Semantic vs Non-Hierarchy Absorption')
    ax.set_xticks(x2)
    ax.set_xticklabels(gpt2_fams)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_gpt2_replication.png', dpi=300)
    plt.close(fig)

    # ======================= SUMMARY JSON =======================
    summary = {
        'task_id': 'statistical_analysis',
        'h1_construct_validity': {
            'description': 'Pearson correlation between first-letter and semantic-hierarchy absorption',
            'excludes_random_sae': {
                'pearson_r': float(r_h1),
                'bootstrap_95_ci_lower': float(ci_l_h1),
                'bootstrap_95_ci_upper': float(ci_u_h1),
                'n_saes': int(main_mask.sum()),
                'supported': bool(r_h1 > 0.6 and ci_l_h1 > 0),
                'rejected': bool(ci_u_h1 < 0.3),
            },
            'includes_random_sae': {
                'pearson_r': float(r_h1_all),
                'bootstrap_95_ci_lower': float(ci_l_h1_all),
                'bootstrap_95_ci_upper': float(ci_u_h1_all),
                'n_saes': len(families),
            }
        },
        'h1_firstletter_vs_nonhierarchy': {
            'pearson_r': float(r_h1_nh),
            'bootstrap_95_ci_lower': float(ci_l_h1_nh),
            'bootstrap_95_ci_upper': float(ci_u_h1_nh),
        },
        'h2_hierarchy_specificity': {
            'description': 'Paired t-test comparing semantic-hierarchy vs non-hierarchy control',
            't_statistic': float(t_h2),
            'p_value': float(p_h2),
            'mean_semantic': float(sem_scores[main_mask].mean()),
            'mean_nonhierarchy': float(nh_scores[main_mask].mean()),
            'supported': bool(sem_scores[main_mask].mean() > nh_scores[main_mask].mean() and p_h2 < 0.05),
        },
        'per_architecture_scores': [
            {
                'family': f,
                'first_letter_absorption': float(fl_lookup[f]['official_absorption_full']),
                'semantic_hierarchy_absorption': float(sem_lookup[f]['mean_hierarchy_absorption']),
                'nonhierarchy_control_absorption': float(nh_lookup[f]['mean_pair_absorption']),
            }
            for f in families
        ],
        'per_hierarchy_probe_auroc': {h: {'mean_auroc': float(np.mean(v)), 'all_aurocs': v}
                                       for h, v in per_hierarchy_auroc.items()},
        'per_hierarchy_absorption_mean': {h: float(np.mean(per_hierarchy_absorption[h]))
                                          for h in hierarchy_names},
        'tau_fs_robustness': tau_robustness,
        'gpt2_replication': gpt2_summary,
        'random_sae_scores': {
            'first_letter': float(fl_lookup['Random']['official_absorption_full']),
            'semantic_hierarchy': float(sem_lookup['Random']['mean_hierarchy_absorption']),
            'nonhierarchy_control': float(nh_lookup['Random']['mean_pair_absorption']),
        },
        'overall_assessment': {
            'h1_construct_validity': 'SUPPORTED' if (r_h1 > 0.6 and ci_l_h1 > 0) else ('REJECTED' if ci_u_h1 < 0.3 else 'INCONCLUSIVE'),
            'h2_hierarchy_specificity': 'SUPPORTED' if (sem_scores[main_mask].mean() > nh_scores[main_mask].mean() and p_h2 < 0.05) else 'REJECTED',
            'h3_robustness': 'SUPPORTED' if all(t['pearson_r_firstletter_semantic'] > 0.5 for t in tau_robustness) else 'INCONCLUSIVE',
        },
        'figures_generated': [
            'fig1_architecture_ranking.png',
            'fig2_firstletter_vs_semantic_scatter.png',
            'fig3_semantic_vs_nonhierarchy.png',
            'fig4_tau_fs_robustness.png',
            'fig5_gpt2_replication.png',
        ],
    }

    with open(OUTPUT_DIR / 'statistical_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Write markdown summary
    md = f"""# Statistical Analysis Summary

## H1: Construct Validity (First-Letter vs Semantic-Hierarchy Absorption)

- **Pearson r** (excluding Random SAE): {r_h1:.3f}
- **Bootstrap 95% CI**: [{ci_l_h1:.3f}, {ci_u_h1:.3f}]
- **Assessment**: {summary['overall_assessment']['h1_construct_validity']}

With Random SAE included:
- Pearson r = {r_h1_all:.3f}, CI = [{ci_l_h1_all:.3f}, {ci_u_h1_all:.3f}]

## H2: Hierarchy Specificity (Semantic vs Non-Hierarchy Control)

- **Paired t-test**: t = {t_h2:.3f}, p = {p_h2:.4f}
- Mean semantic-hierarchy absorption = {sem_scores[main_mask].mean():.3f}
- Mean non-hierarchy control absorption = {nh_scores[main_mask].mean():.3f}
- **Assessment**: {summary['overall_assessment']['h2_hierarchy_specificity']}

## H3: Robustness Across tau_fs

| tau_fs | Pearson r | 95% CI Lower | 95% CI Upper |
|--------|-----------|--------------|--------------|
"""
    for t in tau_robustness:
        md += f"| {t['tau_fs']:.2f} | {t['pearson_r_firstletter_semantic']:.3f} | {t['bootstrap_ci_lower']:.3f} | {t['bootstrap_ci_upper']:.3f} |\n"
    md += f"\n- **Assessment**: {summary['overall_assessment']['h3_robustness']}\n\n"

    md += "## Per-Architecture Scores\n\n"
    md += "| Architecture | First-Letter | Semantic-Hierarchy | Non-Hierarchy Control |\n"
    md += "|--------------|-------------:|-------------------:|----------------------:|\n"
    for entry in summary['per_architecture_scores']:
        md += f"| {entry['family']} | {entry['first_letter_absorption']:.3f} | {entry['semantic_hierarchy_absorption']:.3f} | {entry['nonhierarchy_control_absorption']:.3f} |\n"

    md += "\n## Random-SAE Control\n\n"
    md += f"- First-letter: {summary['random_sae_scores']['first_letter']:.3f}\n"
    md += f"- Semantic-hierarchy: {summary['random_sae_scores']['semantic_hierarchy']:.3f}\n"
    md += f"- Non-hierarchy control: {summary['random_sae_scores']['nonhierarchy_control']:.3f}\n"

    md += "\n## Figures Generated\n\n"
    for fig in summary['figures_generated']:
        md += f"- `{fig}`\n"

    with open(OUTPUT_DIR / 'statistical_analysis_summary.md', 'w') as f:
        f.write(md)

    print("Analysis complete. Outputs saved to:")
    print(f"  {OUTPUT_DIR / 'statistical_analysis_summary.json'}")
    print(f"  {OUTPUT_DIR / 'statistical_analysis_summary.md'}")
    for fig in summary['figures_generated']:
        print(f"  {OUTPUT_DIR / fig}")


if __name__ == '__main__':
    main()
