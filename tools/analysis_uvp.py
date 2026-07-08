import re, ast, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import statsmodels.api as sm
import shap
import matplotlib.pyplot as plt

feature_names = [
    'Actinopterygii',
    'Annelida',
    'Appendicularia',
    'Chaetognatha',
    'Copepoda',
    'Ctenophora',
    'Echinodermata',
    'Eumalacostraca',
    'Mollusca',
    'Ostracoda',
    'Rhizaria',
    'Siphonophorae',
    'Trichodesmium',
    'artefact',
    'detritus',
    'fiber',
    'non appendicularia tunicata',
    'non siphonophorae cnidaria',
    'non_siphonophoran_hydrozoa',
    'other_living'
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
        if delta_max > delta_min:  # avoid div by zero
            df.loc[mask, 'delta'] = 100 * (df.loc[mask, 'delta'] - delta_min) / (delta_max - delta_min)
        else:
            df.loc[mask, 'delta'] = 0.0

    return df

save = True
out_path = r'D:\mojmas\files\Projects\CVPR\plots\results\uvp\test'
filename = r"prediction_uvp.txt"

df = read_df(filename)
P_cols = [f'P{i}' for i in range(20)]

df_radar = df[df['delta'] >= 0]

############## top configs ########################
N = 10
top10_df = df.sort_values(by="accuracy", ascending=False).head(N).reset_index(drop=True)

# Step 2: Extract prototype count columns
proto_matrix = top10_df[P_cols]

# Step 3: Assign meaningful row and column labels
proto_matrix.index = [f'Run {i+1} (acc={a:.2f})' for i, a in enumerate(top10_df['accuracy'])]
proto_matrix.columns = feature_names  # use readable class names

# heatmap
if save:
    # Step 4: Plot heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(proto_matrix, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={'label': 'Prototype Count'})
    # plt.title("Top-10 High-Accuracy Runs: Prototype Allocation Heatmap")
    plt.xlabel("Superclass")
    plt.ylabel("Run (Accuracy)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    out_path_name = out_path + r"\top_10_heatmap_uvp.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()


# radar plot
N = 5
top10_df = df_radar.sort_values(by="accuracy", ascending=False).head(N).reset_index(drop=True)
proto_matrix = top10_df[P_cols]
proto_matrix.columns = feature_names

# Set up angles for radar axes
num_vars = len(feature_names)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the circle

if save:
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)     # start at 12 o'clock
    ax.set_theta_direction(-1)

    num_rings = 4
    ax.set_ylim(0, num_rings)
    ax.set_yticks(range(1, num_rings + 1))
    ax.set_yticklabels([])
    # Plot each run
    for i, row in proto_matrix.iterrows():
        values = row.values.tolist()
        values += values[:1]  # close the radar chart
        ax.plot(angles, values, label=f"Run {i+1} (acc={top10_df.loc[i, 'accuracy']:.2f})", alpha=0.5)
        ax.fill(angles, values, alpha=0.05)

    # Set class labels around the circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=14)
    # ax.set_title("Radar Plot: Prototype Distribution in Top-10 Accuracy Runs", size=14, pad=20)

    ax.set_rlabel_position(0)

    # Optional: Hide y-axis labels or set radial limits
    ax.set_yticklabels([])
    ax.set_rlabel_position(0)

    plt.legend(bbox_to_anchor=(1.3, 1.05), loc='upper left', fontsize=14)
    plt.tight_layout()
    # plt.show()
    out_path_name = out_path + r"\top_5_radar_uvp.pdf"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()


N = 10
top10_df = df_radar.sort_values(by="accuracy", ascending=False).head(N).reset_index(drop=True)
proto_matrix = top10_df[P_cols]
proto_matrix.columns = feature_names

# Set up angles for radar axes
num_vars = len(feature_names)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the circle

if save:
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)     # start at 12 o'clock
    ax.set_theta_direction(-1)

    # Plot each run
    for i, row in proto_matrix.iterrows():
        values = row.values.tolist()
        values += values[:1]  # close the radar chart
        ax.plot(angles, values, label=f"Run {i+1} (acc={top10_df.loc[i, 'accuracy']:.2f})", alpha=0.5)
        ax.fill(angles, values, alpha=0.05)

    # Set class labels around the circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=11)
    # ax.set_title("Radar Plot: Prototype Distribution in Top-10 Accuracy Runs", size=14, pad=20)

    # Optional: Hide y-axis labels or set radial limits
    ax.set_yticklabels([])
    ax.set_rlabel_position(0)

    plt.legend(bbox_to_anchor=(1.3, 1.05), loc='upper left')
    plt.tight_layout()
    # plt.show()
    out_path_name = out_path + r"\top_10_radar_uvp.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()

### Best vs. worst configs

# Define top and bottom N configs
N = 10
df = df.drop(index=70)
df = df.drop(index=4)
topN = df.sort_values(by='accuracy', ascending=False).head(N)
botN = df.sort_values(by='accuracy', ascending=True).head(N)

# Compute average prototype counts per class
top_mean = topN[P_cols].mean().values
bot_mean = botN[P_cols].mean().values

# Build comparison DataFrame
comp_df = pd.DataFrame({
    'Superclass': feature_names,
    'Top Accuracy': top_mean,
    'Bottom Accuracy': bot_mean
})

# Melt for seaborn
comp_df_melted = comp_df.melt(id_vars='Superclass', var_name='Group', value_name='Avg Prototype Count')

if save:
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comp_df_melted, x='Superclass', y='Avg Prototype Count', hue='Group')
    # plt.title(f"Top-{N} vs. Bottom-{N} Prototype Allocation per Class")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    out_path_name = out_path + r"\top_vs_bottom_uvp.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()


# Radar-style prototype shape
top_vals = top_mean.tolist() + [top_mean[0]]
bot_vals = bot_mean.tolist() + [bot_mean[0]]

angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
angles += angles[:1]

if save:
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)     # start at 12 o'clock
    ax.set_theta_direction(-1)

    ax.plot(angles, top_vals, label=f'Top {N} avg', color='green', lw=2)
    ax.fill(angles, top_vals, color='green', alpha=0.1)

    ax.plot(angles, bot_vals, label=f'Bottom {N} avg', color='red', lw=2)
    ax.fill(angles, bot_vals, color='red', alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=11)
    ax.set_yticklabels([])
    # ax.set_title("Radar Plot: Avg Prototype Distribution\nTop vs. Bottom Accuracy Runs", size=13, pad=20)

    yticks = ax.get_yticks()
    yticklabels = [f'{y:.1f}' for y in yticks]
    ax.set_yticklabels(yticklabels, fontsize=14)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=14)
    plt.tight_layout()
    # plt.show()

    out_path_name = out_path + r"\top_vs_bottom_radar_uvp.pdf"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()


# #### Ablation: For each class P0..P19, group rows where that prototype is 1
#
# df_sample = df.sort_values(by="accuracy", ascending=False).head(100)
# # df_sample = df.copy()
#
# impact_named = []
# for i in range(20):
#     be_one = df_sample[df_sample[f'P{i}'] == 1]
#     rest = df_sample[df_sample[f'P{i}'] != 1]
#     if not be_one.empty and not rest.empty:
#         acc_diff = be_one['accuracy'].mean() - rest['accuracy'].mean()
#         impact_named.append((feature_names[i], acc_diff))
#
#
# impact_df_named = pd.DataFrame(impact_named, columns=['Prototype', 'Accuracy gain from having prototype'])
# impact_df_named = impact_df_named.sort_values(by='Accuracy gain from having prototype', ascending=False)
#
#
# if save:
#
#     # Plot
#     plt.figure(figsize=(10, 8))
#     barplot = sns.barplot(
#         data=impact_df_named,
#         y='Prototype',
#         x='Accuracy gain from having prototype',
#         palette='viridis'
#     )
#
#     # Annotate bars with values
#     for p in barplot.patches:
#         value = f"{p.get_width():.2f}"
#         barplot.annotate(value,
#                          (p.get_width(), p.get_y() + p.get_height() / 2),
#                          xytext=(5, 0), textcoords='offset points',
#                          ha='left', va='center', fontsize=9)
#
#     # plt.title("Accuracy Gain from Having Prototypes (per Class)", fontsize=13)
#     plt.xlabel("Accuracy Gain")
#     plt.ylabel("")
#     plt.grid(axis='x', linestyle='--', alpha=0.5)
#     ax = plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     # plt.show()
#     out_path_name = out_path + r"\pro_acc_gain.png"
#     plt.savefig(out_path_name, dpi=600)

# # --- Ridge Regression ---
# ridge_model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0]))
# ridge_model.fit(X, y)
#
# # Extract feature coefficients
# ridge_coef = ridge_model.named_steps['ridgecv'].coef_
# ridge_df = pd.DataFrame({
#     'Superclass': feature_names,
#     'Importance (Ridge Coef)': ridge_coef
# }).sort_values('Importance (Ridge Coef)', ascending=False)
#
# # --- Random Forest Regressor ---
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X, y)
# rf_importance = rf.feature_importances_
#
# rf_df = pd.DataFrame({
#     'Superclass': feature_names,
#     'Importance (Random Forest)': rf_importance
# }).sort_values('Importance (Random Forest)', ascending=False)
#
# # --- Plot both side-by-side ---
# fig, axs = plt.subplots(1, 2, figsize=(16, 6))
#
# # Ridge
# sns.barplot(data=ridge_df, y='Superclass', x='Importance (Ridge Coef)', ax=axs[0], palette='Blues_d')
# axs[0].set_title("Feature Importance (Ridge Regression)")
# axs[0].axvline(0, color='gray', linestyle='--')
# axs[0].grid(axis='x', linestyle='--', alpha=0.3)
#
# # Random Forest
# sns.barplot(data=rf_df, y='Superclass', x='Importance (Random Forest)', ax=axs[1], palette='Greens_d')
# axs[1].set_title("Feature Importance (Random Forest)")
# axs[1].grid(axis='x', linestyle='--', alpha=0.3)
#
# plt.tight_layout()
# # plt.show()