import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

# ================= 配置参数 =================
DATA_DIR = "./hubdata/model"             # CSV 文件所在的文件夹路径
OUTPUT_DIR = "./hubdata/model/plots_output"  # 图片保存的目标文件夹路径（会自动创建）
MODELS = ["RandomForest", "SVM", "XGBoost"]
FOLDS = [1, 2, 3, 4, 5]

# Sorting rules: Set to True if lower values represent better efficacy (e.g., IC50); otherwise False.
ASCENDING_ORDER = True 
# =================================================

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize structures to store metric data
patient_metrics = []
fold_metrics_summary = []  # To store aggregated metrics per fold for the final table

# Structure: { model_name: { real_rank_idx: [predicted_ranks_across_all_patients] } }
rank_envelope_data = {model: {} for model in MODELS}

# 1. Read data and calculate metrics
for model in MODELS:
    for fold in FOLDS:
        file_name = f"results_{model}_fold{fold}.csv"
        file_path = os.path.join(DATA_DIR, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Lists to temporarily store patient metrics for the current fold
        fold_patient_maes = []
        fold_patient_rs = []
        
        # Group by patient (Sample)
        for patient, group in df.groupby("Sample"):
            if len(group) < 2:
                continue  # Skip if there are too few compounds to calculate correlation
            
            actual = group["Normalized"].values
            predicted = group["Predicted"].values
            
            # Calculate MAE
            mae = mean_absolute_error(actual, predicted)
            
            # Calculate Spearman Rank Correlation (R)
            r_spearman, _ = spearmanr(actual, predicted)
            
            # Record basic patient-level metrics
            patient_metrics.append({
                "Model": model,
                "Fold": f"Fold {fold}",
                "Patient": patient,
                "MAE": mae,
                "Spearman_R": r_spearman
            })
            
            fold_patient_maes.append(mae)
            fold_patient_rs.append(r_spearman)
            
            # Process rank data (for Plot 3)
            sorted_indices = np.argsort(actual) if ASCENDING_ORDER else np.argsort(-actual)
            pred_ranks = group["Predicted"].rank(ascending=ASCENDING_ORDER, method="average").values
            
            for real_rank_idx, orig_idx in enumerate(sorted_indices):
                pred_rank = pred_ranks[orig_idx]
                if real_rank_idx not in rank_envelope_data[model]:
                    rank_envelope_data[model][real_rank_idx] = []
                rank_envelope_data[model][real_rank_idx].append(pred_rank)
        
        # Calculate and record average metrics for the current fold
        if fold_patient_maes:
            fold_metrics_summary.append({
                "Model": model,
                "Fold": fold,
                "MAE": np.mean(fold_patient_maes),
                "Spearman_R": np.mean(fold_patient_rs)
            })

# Convert patient metrics to DataFrame
df_metrics = pd.DataFrame(patient_metrics)

if df_metrics.empty:
    print("No valid data found. Please check file paths and formats.")
    exit()

# 修改模型名称以便图表展示更加美观
df_metrics['Model'] = df_metrics['Model'].str.replace('RandomForest', 'Random Forest')
new_models_list = ["Random Forest", "SVM", "XGBoost"]

# ================= Generate and Export CSV Summary =================
df_folds = pd.DataFrame(fold_metrics_summary)
summary_rows = []

for model in MODELS:
    df_model = df_folds[df_folds['Model'] == model]
    if df_model.empty:
        continue
        
    for metric in ['MAE', 'Spearman_R']:
        row = {'Model': model, 'Metric': metric}
        fold_vals = []
        
        # Collect values for Fold 1 to 5
        for fold in FOLDS:
            val_series = df_model[df_model['Fold'] == fold][metric].values
            val = val_series[0] if len(val_series) > 0 else np.nan
            row[f'Fold_{fold}'] = val
            if not np.isnan(val):
                fold_vals.append(val)
        
        # Calculate Mean and Standard Deviation across the folds
        row['Mean'] = np.mean(fold_vals) if fold_vals else np.nan
        row['SD'] = np.std(fold_vals) if fold_vals else np.nan
        summary_rows.append(row)

# Save summary table as CSV
df_summary_output = pd.DataFrame(summary_rows)
csv_output_path = os.path.join(OUTPUT_DIR, "metrics_summary_by_fold.csv")
df_summary_output.to_csv(csv_output_path, index=False)
print(f"Summary table successfully saved to: {os.path.abspath(csv_output_path)}")


# ================= Visualization Settings =================
sns.set_theme(style="white")  # Clean white background without gridlines
plt.rcParams.update({
    'font.sans-serif': 'Arial',
    'axes.unicode_minus': False,
    'font.size': 14,            # Global font size
    'axes.titlesize': 14,       # Axes title size
    'axes.labelsize': 14,       # Axes label size
    'xtick.labelsize': 14,      # X-axis tick size
    'ytick.labelsize': 14,      # Y-axis tick size
    'legend.fontsize': 14,      # Legend font size
    'font.weight': 'normal',    # Regular font weight
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ================= Plot 1 & 2: Combined MAE and Spearman Bar Plots =================
# 合并为 1行2列 的大图，与时间分割测试集的图片格式完全对齐
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Performance across Models (5-Fold CV Patient-Averaged)", y=1.05)

# --- 左图：MAE ---
ax1 = axes[0]
sns.barplot(
    data=df_metrics, x="Model", y="MAE", order=new_models_list,
    ax=ax1, errorbar="sd", capsize=0.1, alpha=0.6, palette="Set2", 
    edgecolor="black", hue="Model", legend=False
)
sns.stripplot(
    data=df_metrics, x="Model", y="MAE", order=new_models_list,
    ax=ax1, color="black", alpha=0.5, jitter=0.2, size=5
)
ax1.set_title("Mean Absolute Error (MAE)")
ax1.set_xlabel("Model Name")
ax1.set_ylabel("MAE Value")

# --- 右图：Spearman_R ---
ax2 = axes[1]
sns.barplot(
    data=df_metrics, x="Model", y="Spearman_R", order=new_models_list,
    ax=ax2, errorbar="sd", capsize=0.1, alpha=0.6, palette="Set2", 
    edgecolor="black", hue="Model", legend=False
)
sns.stripplot(
    data=df_metrics, x="Model", y="Spearman_R", order=new_models_list,
    ax=ax2, color="black", alpha=0.5, jitter=0.2, size=5
)
ax2.set_title("Spearman Correlation")
ax2.set_xlabel("Model Name")
ax2.set_ylabel("Correlation Coefficient")

# 移除顶部和右侧的边框线
sns.despine()
plt.tight_layout()

# 保存合并后的图片
combined_save_path = os.path.join(OUTPUT_DIR, "metric_MAE_Spearman_combined.png")
plt.savefig(combined_save_path, bbox_inches='tight')
plt.close()


# ================= Plot 3: Rank Relationship with Envelope =================
plt.figure(figsize=(10, 6))

colors = {
    "RandomForest": "#1f77b4",  # Blue
    "SVM": "#ff7f0e",           # Orange
    "XGBoost": "#2ca02c"        # Green
}

max_rank_length = 0

for model, rank_dict in rank_envelope_data.items():
    if not rank_dict:
        continue
    
    rank_indices = sorted(rank_dict.keys())
    max_rank_length = max(max_rank_length, len(rank_indices))
    
    means = []
    stds = []
    
    for idx in rank_indices:
        vals = rank_dict[idx]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        
    means = np.array(means)
    stds = np.array(stds)
    xs = np.array(rank_indices) + 1  # Convert to 1-based ranks
    
    display_model = model.replace('RandomForest', 'Random Forest')
    
    # Plot mean prediction line
    plt.plot(xs, means, label=f"{display_model} (Mean)", color=colors[model], linewidth=2.5)
    
    # Calculate envelope boundaries (Mean ± 1 Standard Deviation)
    lower_bound = np.maximum(1, means - stds)
    upper_bound = np.minimum(len(rank_indices), means + stds)
    
    # Fill envelope area
    plt.fill_between(
        xs, lower_bound, upper_bound, 
        color=colors[model], alpha=0.15, 
        label=f"{display_model} (Mean ± SD)"
    )

# Plot standard diagonal reference line
if max_rank_length > 0:
    plt.plot([1, max_rank_length], [1, max_rank_length], color="red", linestyle="--", linewidth=1.5, label="Perfect Rank")

#plt.title("Relationship between Actual and Predicted Drug Efficacy Rank\n(Aggregated across All Patients)")
plt.xlabel("Actual Rank (1 = Top Efficacy)")
plt.ylabel("Predicted Rank (1 = Top Efficacy)")

# Invert Y-axis so that Rank 1 is at the top
plt.gca().invert_yaxis()
plt.grid(False)

# 同样移除第三张图的顶部和右侧边框线
sns.despine()

plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "rank_relationship_envelope.png"), bbox_inches='tight', dpi=450)
plt.close()

print(f"All plots successfully saved to directory: {os.path.abspath(OUTPUT_DIR)}")
