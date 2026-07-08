import re, ast, pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import statsmodels.api as sm
import shap

# ── IEEE figure sizing ────────────────────────────────────────────────────────
MM_TO_INCH = 1 / 25.4
COL1       = 88.9  * MM_TO_INCH   # ~3.50 in — 1-column
COL2       = 181.9 * MM_TO_INCH   # ~7.16 in — 2-column

# ── Global font / style settings ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.titlesize": 9,
    "font.family":      "Arial",
})
sns.set_theme(style="whitegrid")   # called AFTER rcParams


feature_names = [
    'Actinopterygii', 'Annelida', 'Appendicularia', 'Chaetognatha',
    'Copepoda', 'Ctenophora', 'Echinodermata', 'Eumalacostraca',
    'Mollusca', 'Ostracoda', 'Rhizaria', 'Siphonophorae',
    'Trichodesmium', 'artefact', 'detritus', 'fiber',
    'non appendicularia tunicata', 'non siphonophorae cnidaria',
    'non_siphonophoran_hydrozoa', 'other_living'
]


def read_df(filename):
    with open(filename, "r") as f:
        raw = f.read()

    records = []
    for line in raw.strip().splitlines():
        vec_str, acc_str, delta_str = re.match(r'(.*]),\s*(-?[0-9.]+),\s*(-?[0-9.]+)', line).groups()
        vec = ast.literal_eval(vec_str)
        records.append({
            'vec': vec,
            'accuracy': float(acc_str),
            'delta': float(delta_str),
            'total_prototypes': sum(vec)
        })

    df = pd.DataFrame(records)
    P_cols = [f'P{i}' for i in range(20)]
    df[P_cols] = pd.DataFrame(df['vec'].tolist(), index=df.index)

    mask = df['delta'] != -1
    if mask.any():
        delta_min = df.loc[mask, 'delta'].min()
        delta_max = df.loc[mask, 'delta'].max()
        if delta_max > delta_min:
            df.loc[mask, 'delta'] = 100 * (df.loc[mask, 'delta'] - delta_min) / (delta_max - delta_min)
        else:
            df.loc[mask, 'delta'] = 0.0
    return df


save     = True
out_path = r'D:\mojmas\files\Projects\CVPR\plots\results\uvp\test'
df       = read_df(r"prediction_uvp.txt")
P_cols   = [f'P{i}' for i in range(20)]
df_radar = df[df['delta'] >= 0]


# ─────────────────────────────────────────────────────────────────────────────
# Radar helper
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar(df_in, n, out_name, size=COL1 * 1.5):
    top_df       = df_in.sort_values(by="accuracy", ascending=False).head(n).reset_index(drop=True)
    proto_matrix = top_df[P_cols].copy()
    proto_matrix.columns = feature_names

    num_vars = len(feature_names)
    angles   = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles  += angles[:1]

    fig = plt.figure(figsize=(size, size))
    ax  = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    num_rings = 4
    ax.set_ylim(0, num_rings)
    ax.set_yticks(range(1, num_rings + 1))
    ax.set_yticklabels([])

    for i, row in proto_matrix.iterrows():
        values = row.values.tolist() + [row.values[0]]
        ax.plot(angles, values, label=f"Run {i+1} (acc={top_df.loc[i, 'accuracy']:.2f})",
                alpha=0.5, linewidth=0.8)
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=5)
    ax.set_rlabel_position(0)
    plt.legend(bbox_to_anchor=(1.3, 1.05), loc='upper left', fontsize=5)
    plt.savefig(out_path + f"\\{out_name}.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + f"\\{out_name}.png", dpi=600, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Top configs — heatmap
# ─────────────────────────────────────────────────────────────────────────────
N        = 10
top10_df = df.sort_values(by="accuracy", ascending=False).head(N).reset_index(drop=True)

proto_matrix         = top10_df[P_cols].copy()
proto_matrix.index   = [f'Run {i+1} (acc={a:.2f})' for i, a in enumerate(top10_df['accuracy'])]
proto_matrix.columns = feature_names

if save:
    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.45))
    sns.heatmap(proto_matrix, annot=True, fmt="d", cmap="YlOrRd",
                cbar_kws={'label': 'Prototype Count'}, ax=ax,
                annot_kws={"size": 6})
    ax.set_xlabel("Superclass")
    ax.set_ylabel("Run (Accuracy)")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(out_path + r"\top_10_heatmap_uvp.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\top_10_heatmap_uvp.png", dpi=600, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Radar plots
# ─────────────────────────────────────────────────────────────────────────────
if save:
    plot_radar(df_radar, n=5,  out_name="top_5_radar_uvp")
    plot_radar(df_radar, n=10, out_name="top_10_radar_uvp")

# ─────────────────────────────────────────────────────────────────────────────
# Best vs. worst configs
# ─────────────────────────────────────────────────────────────────────────────
N  = 10
df = df.drop(index=70)
df = df.drop(index=4)

topN     = df.sort_values(by='accuracy', ascending=False).head(N)
botN     = df.sort_values(by='accuracy', ascending=True).head(N)
top_mean = topN[P_cols].mean().values
bot_mean = botN[P_cols].mean().values

comp_df = pd.DataFrame({
    'Superclass':      feature_names,
    'Top Accuracy':    top_mean,
    'Bottom Accuracy': bot_mean
})
comp_df_melted = comp_df.melt(id_vars='Superclass', var_name='Group', value_name='Avg Prototype Count')

if save:
    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.45))
    sns.barplot(data=comp_df_melted, x='Superclass', y='Avg Prototype Count',
                hue='Group', ax=ax)
    ax.set_xlabel("Superclass")
    ax.set_ylabel("Avg Prototype Count")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(out_path + r"\top_vs_bottom_uvp.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\top_vs_bottom_uvp.png", dpi=600, bbox_inches='tight')
    plt.close()

# Radar: top vs bottom
top_vals = top_mean.tolist() + [top_mean[0]]
bot_vals = bot_mean.tolist() + [bot_mean[0]]
angles   = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
angles  += angles[:1]

if save:
    size = COL1 * 0.9
    fig  = plt.figure(figsize=(size, size))
    ax   = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, top_vals, label=f'Top {N} avg',    color='green', lw=1.5)
    ax.fill(angles, top_vals, color='green', alpha=0.1)
    ax.plot(angles, bot_vals, label=f'Bottom {N} avg', color='red',   lw=1.5)
    ax.fill(angles, bot_vals, color='red',   alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=5)
    ax.set_yticklabels([])

    yticks      = ax.get_yticks()
    yticklabels = [f'{y:.1f}' for y in yticks]
    ax.set_yticklabels(yticklabels, fontsize=5)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=5)
    plt.savefig(out_path + r"\top_vs_bottom_radar_uvp.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\top_vs_bottom_radar_uvp.png", dpi=600, bbox_inches='tight')
    plt.close()