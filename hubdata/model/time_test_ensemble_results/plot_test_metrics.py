import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

# 配置测试集结果文件夹路径
TEST_OUT_DIR = './hubdata/model/time_test_ensemble_results'

def main():
    print("--------------------------------------------------")
    print("测试集各患者评价指标统计 (Mean ± SD)")
    print("--------------------------------------------------\n")

    models = ['RandomForest', 'SVM', 'XGBoost']
    
    # 存储用于画图的原始数据记录
    plot_data = []

    for model in models:
        csv_path = os.path.join(TEST_OUT_DIR, f'test_results_{model}_ensemble.csv')
        if not os.path.exists(csv_path):
            print(f"找不到模型 {model} 的结果文件，请检查路径。")
            continue

        df = pd.read_csv(csv_path)
        
        # 自动识别真实的药物活性列名
        actual_col = 'Normalized' if 'Normalized' in df.columns else df.columns[3]
        pred_col = 'Predicted'

        patient_spearman = []
        patient_pearson = []
        patient_mae = []

        # 按患者分组计算
        for sample, group in df.groupby('Sample'):
            if len(group) < 2:
                continue
                
            actual = group[actual_col].values
            pred = group[pred_col].values

            s_corr, _ = spearmanr(actual, pred)
            p_corr, _ = pearsonr(actual, pred)
            mae = mean_absolute_error(actual, pred)

            patient_spearman.append(s_corr)
            patient_pearson.append(p_corr)
            patient_mae.append(mae)

        # 统计平均值与标准差
        mean_sp = np.mean(patient_spearman)
        std_sp = np.std(patient_spearman)

        mean_pe = np.mean(patient_pearson)
        std_pe = np.std(patient_pearson)

        mean_mae = np.mean(patient_mae)
        std_mae = np.std(patient_mae)

        # 终端打印结果
        print(f"模型: {model}")
        print(f"Spearman: {mean_sp:.4f} ± {std_sp:.4f}")
        print(f"Pearson : {mean_pe:.4f} ± {std_pe:.4f}")
        print(f"MAE     : {mean_mae:.4f} ± {std_mae:.4f}")
        print("--------------------------------------------------\n")
        
        # 将每个患者的数据存入，用于绘制散点柱状图
        display_model = model.replace('RandomForest', 'Random Forest')
        for sp, m in zip(patient_spearman, patient_mae):
            plot_data.append({'Model': display_model, 'Metric': 'Spearman_R', 'Value': sp})
            plot_data.append({'Model': display_model, 'Metric': 'MAE', 'Value': m})

    if not plot_data:
        print("没有提取到可用于绘图的数据，程序结束。")
        return

    df_plot = pd.DataFrame(plot_data)
    
    # ================= Visualization Settings =================
    # 使用纯白背景，取消网格线
    sns.set_theme(style="white") 
    plt.rcParams.update({
        'font.sans-serif': 'Arial',
        'axes.unicode_minus': False,
        'font.size': 14,            # 全局字体大小统一为14
        'axes.titlesize': 14,       # 标题字体大小
        'axes.labelsize': 14,       # 坐标轴标签大小
        'xtick.labelsize': 14,      # X轴刻度大小
        'ytick.labelsize': 14,      # Y轴刻度大小
        'legend.fontsize': 14,      # 图例字体大小
        'font.weight': 'normal',    # 取消所有加粗
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'figure.dpi': 300,
        'savefig.dpi': 300
    })
    
    # 创建1行2列的并排子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 取消总标题的加粗，统一字号
    fig.suptitle('Performance on Time-Split Test Set (Patient-Averaged)', y=1.05)

    # --- 左图：MAE ---
    ax1 = axes[0]
    df_mae = df_plot[df_plot['Metric'] == 'MAE']
    
    # 绘制底部的柱状图与误差棒
    sns.barplot(
        data=df_mae, x='Model', y='Value', 
        ax=ax1, errorbar="sd", capsize=0.1, 
        alpha=0.6, palette="Set2", edgecolor="black", hue='Model', legend=False
    )
    # 叠加散点（代表每个患者的实际表现）
    sns.stripplot(
        data=df_mae, x='Model', y='Value', 
        ax=ax1, color="black", alpha=0.5, jitter=0.2, size=5
    )
    
    ax1.set_title('Mean Absolute Error (MAE)')
    ax1.set_xlabel('Model Name')
    ax1.set_ylabel('MAE Value')
    ax1.grid(False)

    # --- 右图：Spearman ---
    ax2 = axes[1]
    df_sp = df_plot[df_plot['Metric'] == 'Spearman_R']
    
    sns.barplot(
        data=df_sp, x='Model', y='Value', 
        ax=ax2, errorbar="sd", capsize=0.1, 
        alpha=0.6, palette="Set2", edgecolor="black", hue='Model', legend=False
    )
    sns.stripplot(
        data=df_sp, x='Model', y='Value', 
        ax=ax2, color="black", alpha=0.5, jitter=0.2, size=5
    )
    
    ax2.set_title('Spearman Correlation')
    ax2.set_xlabel('Model Name')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.grid(False)

    # 移除顶部和右侧多余的边框线
    sns.despine()
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(TEST_OUT_DIR, 'test_set_patient_metrics_scatter_bar.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"数据统计与可视化全部完成，带有散点分布的高清图片已保存至:\n{save_path}")

if __name__ == "__main__":
    main()
