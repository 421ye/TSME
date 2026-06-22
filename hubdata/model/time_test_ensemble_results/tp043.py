import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import matplotlib as mpl

# ================= 配置排版参数 =================
# 全局设置 Arial 字体，14.5 号，完全不加粗
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# ================================================

def plot_patient_rankings(csv_file, patient_id):
    # 1. 读取数据
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"找不到文件 {csv_file}，请检查路径。")
        return

    # 2. 筛选特定患者
    df_patient = df[df['Sample'] == patient_id].copy()
    if df_patient.empty:
        print(f"未找到患者 {patient_id} 的数据。")
        return

    # 3. 计算排名与 Spearman 相关系数
    # 逻辑：Normalized / Predicted 数值越小，排名越靠前（1为最高排位）
    df_patient['Real_Rank'] = df_patient['Normalized'].rank(ascending=True, method='first')
    df_patient['Pred_Rank'] = df_patient['Predicted'].rank(ascending=True, method='first')
    
    # 计算 Spearman (可以使用原始值或排名，scipy的spearmanr结果一致)
    spearman_corr, _ = spearmanr(df_patient['Normalized'], df_patient['Predicted'])

    # 4. 按真实排名排序，使得 X 轴呈现从左到右递减的单调直线
    df_patient = df_patient.sort_values(by='Real_Rank').reset_index(drop=True)

    # 5. 开始绘图 (高度为 6，宽度为 7，偏向正方形)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # 绘制真实排名（蓝色实线，圆点）
    ax.plot(
        df_patient.index, 
        df_patient['Real_Rank'], 
        marker='o', 
        linestyle='-', 
        color='royalblue', 
        label='Real Rank'
    )

    # 绘制预测排名（橙色虚线，叉号）
    ax.plot(
        df_patient.index, 
        df_patient['Pred_Rank'], 
        marker='x', 
        linestyle='--', 
        color='darkorange', 
        label='Predicted Rank'
    )

    # 6. 图表细节修饰
    # 反转 Y 轴，让排名 1 (Top Rank) 显示在图片的最上方
    ax.invert_yaxis()
    
    # 轴标签
    ax.set_xlabel('Compounds (Sorted by Real Efficacy Rank)')
    ax.set_ylabel('Rank (1 = Top Rank / Lowest Value)')
    
    # 标题（仅保留 Spearman）
    ax.set_title(f'[SVM_Ensemble] Patient: TTV0021\nSpearman: {spearman_corr:.3f}')
    
    # 去除网格线
    ax.grid(False)
    
    # 图例配置（不加粗，移除边框更清爽）
    ax.legend(frameon=True, edgecolor='lightgray')

    # 紧凑布局防边缘切断
    plt.tight_layout()

    # 保存图片
    output_filename = f'./hubdata/model/time_test_ensemble_results/rank_comparison_TTV0021_RandomForest.png'
    plt.savefig(output_filename, dpi=300)
    print(f"绘图成功！已保存为: {output_filename}")
    
    # 如果在 Jupyter Notebook 中运行，可直接展示
    plt.show()

if __name__ == "__main__":
    # 假设你的文件名为 data.csv，目标患者为 TP_0043
    # 请根据实际情况修改 'data.csv'
    plot_patient_rankings('./hubdata/model/time_test_ensemble_results/test_results_RandomForest_ensemble.csv', 'TP_V021')
