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

    delta_min = df['delta'].min()
    delta_max = df['delta'].max()
    df['delta'] = 100 * (df['delta'] - delta_min) / (delta_max - delta_min) if delta_max > delta_min else 0.0
    return df


feature_names = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers',
    'fruit_and_vegetables', 'household_electrical_devices',
    'household_furniture', 'insects', 'large_carnivores',
    'large_man-made...', 'large_natural_outdoor_scenes',
    'large_omnivores_...', 'medium-sized_mammals',
    'non-insect_invertebrates', 'people', 'reptiles',
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]

save     = True
out_path = r'D:\mojmas\files\Projects\CVPR\plots\results\test'
df       = read_df(r"prediction_cifar.txt")
df_uvp   = read_df(r"prediction_uvp.txt")

df_sample_uvp = df_uvp.sort_values(by="accuracy", ascending=False).head(123)
P_cols = [f'P{i}' for i in range(20)]

# ─────────────────────────────────────────────────────────────────────────────
# Ablation
# ─────────────────────────────────────────────────────────────────────────────
df_sample    = df.sort_values(by="accuracy", ascending=False).head(123)
proto_target = 46
ablation2    = df_sample[np.isclose(df_sample['total_prototypes'], proto_target, atol=5)]

baseline_accuracy              = ablation2.loc[ablation2['accuracy'].idxmin(), 'accuracy']
ablation2['accuracy_improvement'] = ablation2['accuracy'] - baseline_accuracy

sorted_df1 = ablation2.sort_values("delta")
smoothed1  = sm.nonparametric.lowess(sorted_df1["accuracy_improvement"], sorted_df1["delta"], frac=0.3)

# UVP
ablation_uvp                      = df_sample_uvp.copy()
baseline_accuracy                 = ablation_uvp.loc[ablation_uvp['accuracy'].idxmin(), 'accuracy']
ablation_uvp['accuracy_improvement'] = ablation_uvp['accuracy'] - baseline_accuracy

sorted_df_uvp = ablation_uvp.sort_values("delta")
smoothed3     = sm.nonparametric.lowess(sorted_df_uvp["accuracy_improvement"], sorted_df_uvp["delta"], frac=0.4)

# Normalize UVP to CIFAR scale
y1, y3    = smoothed1[:, 0], smoothed3[:, 0]
min1, max1 = np.min(y1), np.max(y1)
min3, max3 = np.min(y3), np.max(y3)
smoothed3_norm       = smoothed3.copy()
smoothed3_norm[:, 0] = (y3 - min3) / (max3 - min3) * (max1 - min1) + min1

# Vary total_prototypes
top_10       = df.nlargest(50, 'accuracy')
bottom_10    = df.nsmallest(5, 'accuracy')
specific_row = df.loc[[114, 113, 111, 110]]
df_sample    = pd.concat([top_10, bottom_10, specific_row]).reset_index(drop=True)

sorted_df2                         = df_sample.sort_values("total_prototypes")
sorted_df2['accuracy_improvement'] = sorted_df2['accuracy'] - 50.57
smoothed2 = sm.nonparametric.lowess(sorted_df2["accuracy_improvement"], sorted_df2["total_prototypes"], frac=0.3)

# UVP
sorted_df2_uvp                         = df_sample_uvp.sort_values("total_prototypes")
sorted_df2_uvp['accuracy_improvement'] = sorted_df2_uvp['accuracy'] - 44.07
smoothed4 = sm.nonparametric.lowess(sorted_df2_uvp["accuracy_improvement"], sorted_df2_uvp["total_prototypes"], frac=0.3)

if save:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.45), sharey=True)

    ax1.plot(smoothed1[:, 0], smoothed1[:, 1], color=plt.cm.Blues(0.6),
             linestyle='-', marker='o', markersize=3, linewidth=1, label='LOWESS – CIFAR')
    ax1.plot(smoothed3_norm[:, 0], smoothed3_norm[:, 1], color=plt.cm.Blues(0.9),
             linestyle='--', marker='s', markersize=3, linewidth=1, label='LOWESS – UVP6NET')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Delta")
    ax1.set_ylabel("Accuracy Improvement (from baseline)")
    ax1.legend()
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(smoothed2[:, 0], smoothed2[:, 1], color=plt.cm.Reds(0.6),
             linestyle='-', marker='^', markersize=3, linewidth=1, label='LOWESS – CIFAR')
    ax2.plot(smoothed4[:, 0], smoothed4[:, 1], color=plt.cm.Reds(0.9),
             linestyle='--', marker='d', markersize=3, linewidth=1, label='LOWESS – UVP6NET')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel("Total Number of Prototypes")
    ax2.legend()
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.subplots_adjust(wspace=0.05)
    plt.savefig(out_path + r"\compare_prototypes_vs_delta_twodataset.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\compare_prototypes_vs_delta_twodataset.png", dpi=600, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SHAP Analysis
# ─────────────────────────────────────────────────────────────────────────────
df_sample = df.copy()
X = df_sample[P_cols]
y = df_sample['accuracy']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

explainer   = shap.Explainer(rf)
shap_values = explainer(X)

if save:
    plt.figure(figsize=(COL2, COL2 * 0.6))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_size=None, show=False)
    plt.savefig(out_path + r"\shap_summary_dot.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\shap_summary_dot.png", dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(COL2, COL2 * 0.6))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type='bar', plot_size=None, show=False)
    plt.savefig(out_path + r"\shap_bar.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\shap_bar.png", dpi=600, bbox_inches='tight')
    plt.close()

if save:
    plt.figure(figsize=(COL2, COL2 * 0.6))
    highlight_mask = (X == 1)
    for i, name in enumerate(feature_names):
        idxs      = np.where(highlight_mask.iloc[:, i])[0]
        shap_vals = shap_values.values[idxs, i]
        plt.scatter([shap_vals], [np.full_like(shap_vals, i)],
                    color='green', alpha=0.5, s=8,
                    label='Prototype = 1' if i == 0 else "")

    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_size=None, show=False)
    plt.legend()
    plt.savefig(out_path + r"\shap_summary_dot_prototype_1.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\shap_summary_dot_prototype_1.png", dpi=600, bbox_inches='tight')
    plt.close()

if save:
    X_binary      = (X == 1).astype(int)
    rf_bin        = RandomForestRegressor().fit(X_binary, y)
    explainer_bin = shap.Explainer(rf_bin)
    shap_vals_bin = explainer_bin(X_binary)

    plt.figure(figsize=(COL2, COL2 * 0.6))
    shap.summary_plot(shap_vals_bin, X_binary, feature_names=feature_names,
                      plot_size=None, show=False)
    plt.savefig(out_path + r"\shap_summary_binary.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\shap_summary_binary.png", dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(COL2, COL2 * 0.6))
    shap.summary_plot(shap_vals_bin, X_binary, feature_names=feature_names,
                      plot_type='bar', plot_size=None, show=False)
    plt.savefig(out_path + r"\shap_bar_binary.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\shap_bar_binary.png", dpi=600, bbox_inches='tight')
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
    plt.savefig(out_path + r"\top_10_heatmap.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\top_10_heatmap.png", dpi=600, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Radar plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar(df_in, n, out_name, size=COL1):
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

    for i, row in proto_matrix.iterrows():
        values = row.values.tolist() + [row.values[0]]
        ax.plot(angles, values, label=f"Run {i+1} (acc={top_df.loc[i, 'accuracy']:.2f})",
                alpha=0.5, linewidth=0.8)
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=5)
    ax.set_yticklabels([])
    ax.set_rlabel_position(0)
    plt.legend(bbox_to_anchor=(1.3, 1.05), loc='upper left', fontsize=5)
    plt.savefig(out_path + f"\\{out_name}.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + f"\\{out_name}.png", dpi=600, bbox_inches='tight')
    plt.close()

if save:
    plot_radar(df, n=5,  out_name="top_5_radar",  size=COL1 * 1.5)
    plot_radar(df, n=10, out_name="top_10_radar", size=COL1 * 1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Best vs. worst configs
# ─────────────────────────────────────────────────────────────────────────────
N        = 10
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
    plt.savefig(out_path + r"\top_vs_bottom.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\top_vs_bottom.png", dpi=600, bbox_inches='tight')
    plt.close()

# Radar: top vs bottom
top_vals = top_mean.tolist() + [top_mean[0]]
bot_vals = bot_mean.tolist() + [bot_mean[0]]
angles   = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
angles  += angles[:1]

if save:
    size = COL1 * 1.5
    fig  = plt.figure(figsize=(size, size))
    ax   = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, top_vals, label=f'Top {N} avg',    color='green', lw=1.5)
    ax.fill(angles, top_vals, color='green', alpha=0.1)
    ax.plot(angles, bot_vals, label=f'Bottom {N} avg', color='red',   lw=1.5)
    ax.fill(angles, bot_vals, color='red',   alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)   # size from rcParams
    ax.set_yticklabels([])

    yticks      = ax.get_yticks()
    yticklabels = [f'{y:.1f}' for y in yticks]
    ax.set_yticklabels(yticklabels)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig(out_path + r"\top_vs_bottom_radar.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_path + r"\top_vs_bottom_radar.png", dpi=600, bbox_inches='tight')
    plt.close()