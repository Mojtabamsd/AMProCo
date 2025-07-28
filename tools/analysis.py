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
    'aquatic_mammals',
    'fish',
    'flowers',
    'food_containers',
    'fruit_and_vegetables',
    'household_electrical_devices',
    'household_furniture',
    'insects',
    'large_carnivores',
    'large_man-made_outdoor_things',
    'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores',
    'medium-sized_mammals',
    'non-insect_invertebrates',
    'people',
    'reptiles',
    'small_mammals',
    'trees',
    'vehicles_1',
    'vehicles_2'
]

save = True
out_path = r'D:\mojmas\files\Projects\CVPR\plots\results'
filename = r"prediction2.txt"
with open(filename, "r") as f:
    raw = f.read()

records = []
for line in raw.strip().splitlines():
    vec_str, acc_str, delta_str = re.match(r'(.*]),\s*([0-9.]+),\s*([0-9.]+)', line).groups()
    vec = ast.literal_eval(vec_str)
    records.append({'vec': vec,
                    'accuracy': float(acc_str),
                    'delta': float(delta_str),
                    'total_prototypes': sum(vec)})

df = pd.DataFrame(records)

P_cols = [f'P{i}' for i in range(20)]
df[P_cols] = pd.DataFrame(df['vec'].tolist(), index=df.index)

delta_min = df['delta'].min()
delta_max = df['delta'].max()
df['delta'] = 100 * (df['delta'] - delta_min) / (delta_max - delta_min)


############### Ablation ########################
# 1 Fix total_prototypes, vary delta
df_sample = df.sort_values(by="accuracy", ascending=False).head(123)

proto_target = df_sample['total_prototypes'].mode()[0]
proto_target = 46
# ablation2 = df[df['total_prototypes'] == proto_target]
ablation2 = df_sample[np.isclose(df_sample['total_prototypes'], proto_target, atol=5)]

baseline_accuracy = ablation2.loc[ablation2['accuracy'].idxmin(), 'accuracy']
# baseline_accuracy = 50.57
ablation2['accuracy_improvement'] = ablation2['accuracy'] - baseline_accuracy

sorted_df1 = ablation2.sort_values("delta")
lowess1 = sm.nonparametric.lowess
smoothed1 = lowess1(sorted_df1["accuracy_improvement"], sorted_df1["delta"], frac=0.3)

# 2 vary total_prototypes
top_10 = df.nlargest(50, 'accuracy')
bottom_10 = df.nsmallest(5, 'accuracy')
specific_row = df.loc[[114, 113, 111, 110]]

# Combine them
df_sample = pd.concat([top_10, bottom_10, specific_row])

# Optional: reset index if needed
df_sample = df_sample.reset_index(drop=True)
# df_sample = df.copy()

sorted_df2 = df_sample.sort_values("total_prototypes")

# baseline_accuracy = sorted_df.loc[sorted_df['accuracy'].idxmin(), 'accuracy']
baseline_accuracy = 50.57 # when all is one
sorted_df2['accuracy_improvement'] = sorted_df2['accuracy'] - baseline_accuracy

lowess2 = sm.nonparametric.lowess
smoothed2 = lowess2(sorted_df2["accuracy_improvement"], sorted_df2["total_prototypes"], frac=0.3)


if save:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # --- First subplot:
    ax1.plot(smoothed1[:, 0], smoothed1[:, 1], color='blue', label='LOWESS trend')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel("Delta")
    ax1.set_ylabel("Accuracy Improvement (from baseline)")
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Second subplot:
    ax2.plot(smoothed2[:, 0], smoothed2[:, 1], color='red', label='LOWESS trend')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel("Total Number of Prototypes")
    # Y-label is omitted here since it's shared from ax1
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Tight layout and save
    plt.tight_layout()
    out_path_name = out_path + r"\compare_prototypes_vs_delta.png"
    plt.savefig(out_path_name, dpi=600)
    plt.close()

############### SHAP Analysis ########################
# feature importance to answer "Which classes' prototype counts explain accuracy best?"
# df_sample = df.sort_values(by="accuracy", ascending=False).head(100)
df_sample = df.copy()

X = df_sample[P_cols]  # P0..P19
y = df_sample['accuracy']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Use TreeExplainer for RF
explainer = shap.Explainer(rf)
shap_values = explainer(X)

if save:
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_size=(12, 8), show=False)
    out_path_name = out_path + r"\shap_summary_dot.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar', plot_size=(12, 8), show=False)
    out_path_name = out_path + r"\shap_bar.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()


if save:
    plt.figure()
    highlight_mask = (X == 1)

    # Loop through features to overlay red dots where prototype == 1
    for i, name in enumerate(feature_names):
        idxs = np.where(highlight_mask.iloc[:, i])[0]
        shap_vals = shap_values.values[idxs, i]

        plt.scatter(
            [shap_vals],
            [np.full_like(shap_vals, i)],
            color='green',
            alpha=0.5,
            s=15,
            label='Prototype = 1' if i == 0 else ""
        )

    # Then add the normal summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_size=(12, 8), show=False)
    plt.legend()
    # plt.title("Red = SHAP values where prototype count == 1")
    plt.tight_layout()
    # plt.show()
    out_path_name = out_path + r"\shap_summary_dot_prototype_1.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()

if save:
    plt.figure()
    # binary, 1 again rest
    X_binary = (X == 1).astype(int)  # 1 = exactly 1 prototype; 0 = all others
    rf_bin = RandomForestRegressor().fit(X_binary, y)
    explainer_bin = shap.Explainer(rf_bin)
    shap_vals_bin = explainer_bin(X_binary)

    shap.summary_plot(shap_vals_bin, X_binary, feature_names=feature_names, plot_size=(12, 8), show=False)
    out_path_name = out_path + r"\shap_summary_binary.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_vals_bin, X_binary, feature_names=feature_names, plot_type='bar', plot_size=(12, 8), show=False)
    out_path_name = out_path + r"\shap_bar_binary.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()

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
    out_path_name = out_path + r"\top_10_heatmap.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()


# radar plot
N = 5
top10_df = df.sort_values(by="accuracy", ascending=False).head(N).reset_index(drop=True)
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
    # Plot each run
    for i, row in proto_matrix.iterrows():
        values = row.values.tolist()
        values += values[:1]  # close the radar chart
        ax.plot(angles, values, label=f"Run {i+1} (acc={top10_df.loc[i, 'accuracy']:.2f})", alpha=0.5)
        ax.fill(angles, values, alpha=0.05)

    # Set class labels around the circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    # ax.set_title("Radar Plot: Prototype Distribution in Top-10 Accuracy Runs", size=14, pad=20)

    # Optional: Hide y-axis labels or set radial limits
    ax.set_yticklabels([])
    ax.set_rlabel_position(0)

    plt.legend(bbox_to_anchor=(1.3, 1.05), loc='upper left')
    plt.tight_layout()
    # plt.show()
    out_path_name = out_path + r"\top_5_radar.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()


N = 10
top10_df = df.sort_values(by="accuracy", ascending=False).head(N).reset_index(drop=True)
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
    # Plot each run
    for i, row in proto_matrix.iterrows():
        values = row.values.tolist()
        values += values[:1]  # close the radar chart
        ax.plot(angles, values, label=f"Run {i+1} (acc={top10_df.loc[i, 'accuracy']:.2f})", alpha=0.5)
        ax.fill(angles, values, alpha=0.05)

    # Set class labels around the circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    # ax.set_title("Radar Plot: Prototype Distribution in Top-10 Accuracy Runs", size=14, pad=20)

    # Optional: Hide y-axis labels or set radial limits
    ax.set_yticklabels([])
    ax.set_rlabel_position(0)

    plt.legend(bbox_to_anchor=(1.3, 1.05), loc='upper left')
    plt.tight_layout()
    # plt.show()
    out_path_name = out_path + r"\top_10_radar.png"
    plt.savefig(out_path_name, dpi=600, bbox_inches='tight')
    plt.close()

### Best vs. worst configs

# Define top and bottom N configs
N = 10
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
    out_path_name = out_path + r"\top_vs_bottom.png"
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

    ax.plot(angles, top_vals, label=f'Top {N} avg', color='green', lw=2)
    ax.fill(angles, top_vals, color='green', alpha=0.1)

    ax.plot(angles, bot_vals, label=f'Bottom {N} avg', color='red', lw=2)
    ax.fill(angles, bot_vals, color='red', alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_yticklabels([])
    # ax.set_title("Radar Plot: Avg Prototype Distribution\nTop vs. Bottom Accuracy Runs", size=13, pad=20)

    yticks = ax.get_yticks()
    yticklabels = [f'{y:.1f}' for y in yticks]
    ax.set_yticklabels(yticklabels, fontsize=9)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    # plt.show()

    out_path_name = out_path + r"\top_vs_bottom_radar.png"
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