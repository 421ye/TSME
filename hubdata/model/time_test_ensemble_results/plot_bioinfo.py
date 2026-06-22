import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Point directly to one of the paths you found with fc-list
font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
font_prop = fm.FontProperties(fname=font_path)


# 尝试导入智能排版库
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("提示：检测到未安装 adjustText 库，文本防重叠功能将受限。建议在终端运行 'pip install adjustText'。")

warnings.filterwarnings("ignore")

# ================= 配置参数 =================
MODEL_DIR = './hubdata/model'
TEST_OUT_DIR = os.path.join(MODEL_DIR, 'time_test_ensemble_results')
GENE2IND_FILE = './2026_04_ds/dataset_01_w_24_compounds/gene2ind.txt'
GENE_EMBED_FILE = './2026_04_ds/dataset_01_w_24_compounds/all_gene_embedding.csv'
ANALYSIS_OUT_DIR = os.path.join(MODEL_DIR, 'xgboost_bioinfo_analysis_full_cohort')

os.makedirs(ANALYSIS_OUT_DIR, exist_ok=True)
# ============================================

def main():
    print("--------------------------------------------------")
    print("提取全队列 (5-Fold 验证集 + 测试集) 预测结果...")
    print("--------------------------------------------------")

    patient_performance = []

    # 1. 提取 5-Fold 交叉验证中每个患者的折外预测性能
    for fold in range(1, 6):
        csv_path = os.path.join(MODEL_DIR, f'results_XGBoost_fold{fold}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            actual_col = 'Normalized' if 'Normalized' in df.columns else df.columns[3]
            for sample, group in df.groupby('Sample'):
                if len(group) < 2:
                    continue
                r, _ = spearmanr(group[actual_col], group['Predicted'])
                patient_performance.append({'Sample': sample, 'Spearman_R': r, 'Cohort': f'CV_Fold{fold}'})

    # 2. 提取时间分割测试集的预测性能
    test_csv = os.path.join(TEST_OUT_DIR, 'test_results_XGBoost_ensemble.csv')
    if os.path.exists(test_csv):
        df_test = pd.read_csv(test_csv)
        actual_col = 'Normalized' if 'Normalized' in df_test.columns else df_test.columns[3]
        for sample, group in df_test.groupby('Sample'):
            if len(group) < 2:
                continue
            r, _ = spearmanr(group[actual_col], group['Predicted'])
            patient_performance.append({'Sample': sample, 'Spearman_R': r, 'Cohort': 'TestSet'})

    df_perf = pd.DataFrame(patient_performance)
    
    # 去重处理
    df_perf = df_perf.drop_duplicates(subset=['Sample'], keep='first')
    
    # 3. 读取 Gene 字典并解析名称
    gene_dict = {}
    with open(GENE2IND_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                gene_dict[int(parts[0])] = parts[1]
                
    num_genes = max(gene_dict.keys()) + 1
    gene_names = [gene_dict.get(i, f"Gene_{i}") for i in range(num_genes)]

    # 4. 读取 Gene Embedding 特征矩阵
    df_embed = pd.read_csv(GENE_EMBED_FILE, header=None)
    df_embed.rename(columns={0: 'Sample'}, inplace=True)
    
    col_mapping = {i+1: gene_names[i] for i in range(len(gene_names)) if (i+1) in df_embed.columns}
    df_embed.rename(columns=col_mapping, inplace=True)
    
    # 5. 合并性能与基因数据，并保存全量源数据
    df_merged = pd.merge(df_perf, df_embed, on='Sample', how='inner')
    df_merged.to_csv(os.path.join(ANALYSIS_OUT_DIR, "source_data_all_features_and_performance.csv"), index=False)
    print("已保存：全量源数据 (Features + Performance)")

    # 6. 生信差异分析
    analysis_results = []
    gene_cols = [c for c in df_merged.columns if c not in ['Sample', 'Spearman_R', 'Cohort']]
    
    for gene in gene_cols:
        group_1 = df_merged[df_merged[gene] == 1]['Spearman_R'].dropna()
        group_0 = df_merged[df_merged[gene] == 0]['Spearman_R'].dropna()
        
        if len(group_1) < 5 or len(group_0) < 5:
            continue
            
        mean_1 = group_1.mean()
        mean_0 = group_0.mean()
        delta = mean_1 - mean_0
        
        stat, p_val = mannwhitneyu(group_1, group_0, alternative='two-sided')
        p_val = max(p_val, 1e-15) 
        
        analysis_results.append({
            'Gene': gene,
            'Delta_R': delta,
            'P_Value': p_val,
            'Neg_Log10_P': -np.log10(p_val)
        })
        
    df_res = pd.DataFrame(analysis_results)
    if df_res.empty:
        return
        
    # 保存分析结果与火山图源数据
    df_res.to_csv(os.path.join(ANALYSIS_OUT_DIR, "xgboost_gene_differential_analysis.csv"), index=False)
    df_res.to_csv(os.path.join(ANALYSIS_OUT_DIR, "source_data_volcano_plot.csv"), index=False)
    print("已保存：差异分析结果与火山图源数据")

    sig_positive = df_res[(df_res['P_Value'] < 0.05) & (df_res['Delta_R'] > 0)].sort_values(by='P_Value')
    sig_negative = df_res[(df_res['P_Value'] < 0.05) & (df_res['Delta_R'] < 0)].sort_values(by='P_Value')
    
    top_pos_genes = sig_positive['Gene'].head(10).tolist()
    top_neg_genes = sig_negative['Gene'].head(10).tolist()

    # ================= 7. 生信高级可视化 =================
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        'font.sans-serif': 'Arial', 
        'axes.unicode_minus': False,
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.weight': 'normal',
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'figure.dpi': 300,
        'savefig.dpi': 300
    })

    fig = plt.figure(figsize=(16, 8))
    
    # --- 子图 A: 火山图 ---
    ax1 = plt.subplot(1, 2, 1)
    
    ax1.scatter(df_res['Delta_R'], df_res['Neg_Log10_P'], color='#E0E0E0', alpha=0.6, s=30, label='Non-significant')
    ax1.scatter(sig_positive['Delta_R'], sig_positive['Neg_Log10_P'], color='#D95F02', alpha=0.8, s=50, label='Enhances Prediction (p < 0.05)')
    ax1.scatter(sig_negative['Delta_R'], sig_negative['Neg_Log10_P'], color='#377EB8', alpha=0.8, s=50, label='Degrades Prediction (p < 0.05)')

    ax1.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 【修复重点 1】：动态扩展坐标轴范围，给文字留出呼吸空间，绝对防止切边
    x_min, x_max = ax1.get_xlim()
    ax1.set_xlim(x_min - (x_max - x_min) * 0.15, x_max + (x_max - x_min) * 0.15)
    y_min, y_max = ax1.get_ylim()
    ax1.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)

    # 【修复重点 2】：收集文本对象并交由 adjust_text 智能排版
    texts = []
    # 标注 Top 4 正向基因
    for i, row in sig_positive.head(4).iterrows():
        texts.append(ax1.text(row['Delta_R'], row['Neg_Log10_P'], row['Gene'], fontsize=14, color='black'))
    # 标注 Top 4 负向基因
    for i, row in sig_negative.head(4).iterrows():
        texts.append(ax1.text(row['Delta_R'], row['Neg_Log10_P'], row['Gene'], fontsize=14, color='black'))

    if HAS_ADJUST_TEXT and len(texts) > 0:
        adjust_text(texts, ax=ax1,
                    arrowprops=dict(arrowstyle="-", color='gray', lw=1.0, alpha=0.8),
                    expand_points=(1.5, 1.5), 
                    expand_text=(1.2, 1.2))

    ax1.set_title('Volcano Plot: Gene Impact on Predictability (Full Cohort)')
    ax1.set_xlabel('Performance Change (Delta Spearman R)')
    ax1.set_ylabel('-log10(P-value)')
    ax1.legend(loc='upper right', frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- 子图 B: 突变富集热图 ---
    ax2 = plt.subplot(1, 2, 2)
    
    df_sorted = df_merged.sort_values(by='Spearman_R', ascending=False)
    
    if len(top_pos_genes) > 0 or len(top_neg_genes) > 0:
        top_genes_list = top_pos_genes + top_neg_genes
        heatmap_df = df_sorted[top_genes_list].copy()
        
        rename_dict = {g: f"{g} (+)" for g in top_pos_genes}
        rename_dict.update({g: f"{g} (-)" for g in top_neg_genes})
        heatmap_df.rename(columns=rename_dict, inplace=True)
        
        heatmap_data = heatmap_df.T
        
        # 保存热图源数据
        heatmap_data.to_csv(os.path.join(ANALYSIS_OUT_DIR, "source_data_heatmap.csv"))
        print("已保存：热图源数据矩阵")

        cmap = sns.color_palette(["#F0F0F0", "#1B9E77"])
        
        sns.heatmap(
            heatmap_data, 
            cmap=cmap, 
            cbar=False, 
            yticklabels=True, 
            xticklabels=False,
            linewidths=0.5,
            linecolor='white',
            ax=ax2
        )
        
        ax2.set_title('Significant Features in Patients (Sorted by XGBoost Accuracy ->)')
        ax2.set_xlabel('Patients (High Accuracy to Low Accuracy)')
        ax2.set_ylabel('Top Significant Genes (+ Enhances, - Degrades)')

    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_OUT_DIR, 'xgboost_bioinformatics_analysis_full.png')
    plt.savefig(save_path, bbox_inches='tight')
    
    print(f"生信图表已重绘完毕，完美解决标签重叠问题，已保存至:\n{save_path}")

if __name__ == "__main__":
    main()
