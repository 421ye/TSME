import csv
import os
import logging
import yaml
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 评价指标
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau

# 传统 ML 库
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

# 引入xgboost
import xgboost as xgb

# 过滤SVM的未收敛警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==========================================
# 核心训练与评估类 (Pure Machine Learning)
# ==========================================
class MLPipeline:
    def __init__(self, root: str, yaml_path: str) -> None:
        self.conf = {}
        with open(yaml_path, 'r') as f:
            self.conf.update(yaml.safe_load(f))
            
        hub_root = self.conf.get('hub', {}).get('root', './hubdata')
        model_repo = self.conf.get('hub', {}).get('model', {}).get('repo', 'model')
        dataset_repo = self.conf.get('hub', {}).get('dataset', {}).get('repo', 'ds')
        
        self.model_dir = os.path.join(hub_root, model_repo)
        self.dataset_dir = os.path.join(hub_root, dataset_repo)
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.setup_logger()

    def setup_logger(self):
        self.logger = logging.getLogger('ml.pipeline')
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        self.logger.setLevel(getattr(logging, self.conf.get("log", {}).get("level", "INFO").upper()))
        formatter = logging.Formatter('[LOG] %(asctime)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        fh = logging.FileHandler(os.path.join(self.model_dir, 'ml_training.log'), mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def calc_metrics(self, y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            pearson, _ = pearsonr(y_true, y_pred)      
            spearman, _ = spearmanr(y_true, y_pred)    
            kendall, _ = kendalltau(y_true, y_pred)    
        else:
            pearson, spearman, kendall = np.nan, np.nan, np.nan
            
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 
                'Pearson': pearson, 'Spearman': spearman, 'Kendall': kendall}

    # ---------------- 特征工程模块 ----------------
    def _smiles_to_morgan(self, smiles, radius=2, nBits=1024):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((nBits,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp, dtype=np.float32)

    def _load_ml_data(self, data_dir):
        gene_dict = {}
        gene_file = os.path.join(data_dir, "all_gene_embedding.csv")
        if not os.path.exists(gene_file):
            return np.array([]), np.array([]), pd.DataFrame()
            
        with open(gene_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                sample_id = row[0].strip().replace('\ufeff', '')
                features = np.array([float(x.strip()) for x in row[1:]], dtype=np.float32)
                gene_dict[sample_id] = features

        X_list, y_list = [], []
        df_label = pd.read_csv(os.path.join(data_dir, "all_dataset_Sensitivity_labeled.csv"))
        
        for _, row in df_label.iterrows():
            sample_id = str(row.iloc[0]).strip().replace('\ufeff', '')
            smiles = str(row.iloc[1]).strip()
            target = float(row.iloc[2]) 
            
            gene_feat = gene_dict.get(sample_id)
            if gene_feat is None:
                continue
            morgan_feat = self._smiles_to_morgan(smiles)
            
            concat_feat = np.concatenate([gene_feat, morgan_feat])
            X_list.append(concat_feat)
            y_list.append(target)
            
        return np.array(X_list), np.array(y_list), df_label

    # ---------------- 训练与验证 ----------------
    def run_ml_baselines(self, train_dir, val_dir, fold):
        self.logger.info(f"--- 提取特征中 (Fold {fold}) ---")
        X_train, y_train, _ = self._load_ml_data(train_dir)
        X_val, y_val, df_val = self._load_ml_data(val_dir)
        
        if len(X_train) == 0 or len(X_val) == 0:
            self.logger.warning(f"Fold {fold} 数据加载失败，跳过。")
            return

        # 初始化并训练标准化器 (Scaler)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 保存这个 Fold 的 Scaler (SVM在测试集必须依赖它)
        scaler_path = os.path.join(self.model_dir, f'ml_scaler_fold{fold}.pkl')
        joblib.dump(scaler, scaler_path)
        self.logger.info(f"Scaler saved to {scaler_path}")

        # 定义三种传统模型
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
            'SVM': SVR(kernel='rbf', C=1.0, epsilon=0.1, max_iter=500),
            'XGBoost': xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
        }

        for model_name, model in models.items():
            self.logger.info(f"开始训练 {model_name} (Fold {fold}) ...")
            
            fig_dir = os.path.join(self.model_dir, 'fig', model_name, f'fold_{fold}')
            os.makedirs(fig_dir, exist_ok=True)
            
            # --- 训练与提取Loss ---
            if model_name == 'SVM':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                train_pred = model.predict(X_train_scaled)
                
            elif model_name == 'XGBoost':
                model.fit(
                    X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_val, y_val)], 
                    verbose=False
                )
                y_pred = model.predict(X_val)
                train_pred = model.predict(X_train)
                
                # 记录 XGBoost 迭代 Loss
                results = model.evals_result()
                xgb_train_loss = [x**2 for x in results['validation_0']['rmse']]
                xgb_val_loss = [x**2 for x in results['validation_1']['rmse']]
                
                xgb_step_csv = os.path.join(fig_dir, f'xgboost_step_loss_fold{fold}.csv')
                with open(xgb_step_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Iteration(Tree)", "Train_Loss(MSE)", "Val_Loss(MSE)"])
                    for i in range(len(xgb_train_loss)):
                        writer.writerow([i + 1, f"{xgb_train_loss[i]:.6f}", f"{xgb_val_loss[i]:.6f}"])
                
            else: # Random Forest
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                train_pred = model.predict(X_train)

            # 保存训练好的模型权重
            model_save_path = os.path.join(self.model_dir, f'{model_name}_fold{fold}.pkl')
            joblib.dump(model, model_save_path)
            self.logger.info(f"[{model_name}] Model weights saved to {model_save_path}")
                
            # 计算最终指标
            train_metrics = self.calc_metrics(y_train, train_pred)
            val_metrics = self.calc_metrics(y_val, y_pred)
            
            train_loss_final = train_metrics['RMSE']**2
            val_loss_final = val_metrics['RMSE']**2
            
            self.logger.info(f"[{model_name}] Val MSE: {val_loss_final:.4f} | Pearson(Val): {val_metrics['Pearson']:.4f} | Sp(Rank): {val_metrics['Spearman']:.4f}")
            
            self.log_fold_summary(model_name, fold, train_metrics, val_metrics)
            
            # 保存汇总指标 CSV
            ml_metrics_csv = os.path.join(fig_dir, f'loss_metrics_fold{fold}.csv')
            with open(ml_metrics_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Phase", "Loss(MSE)", "MAE", "RMSE", "R2", "Pearson", "Spearman", "Kendall"])
                writer.writerow(["Train", f"{train_loss_final:.6f}", f"{train_metrics['MAE']:.6f}", f"{train_metrics['RMSE']:.6f}", f"{train_metrics['R2']:.6f}", f"{train_metrics['Pearson']:.6f}", f"{train_metrics['Spearman']:.6f}", f"{train_metrics['Kendall']:.6f}"])
                writer.writerow(["Val", f"{val_loss_final:.6f}", f"{val_metrics['MAE']:.6f}", f"{val_metrics['RMSE']:.6f}", f"{val_metrics['R2']:.6f}", f"{val_metrics['Pearson']:.6f}", f"{val_metrics['Spearman']:.6f}", f"{val_metrics['Kendall']:.6f}"])

            # 保存预测结果 CSV
            df_res = df_val.copy()
            df_res['Predicted'] = y_pred
            out_file = os.path.join(self.model_dir, f"results_{model_name}_fold{fold}.csv")
            df_res.to_csv(out_file, index=False)
            
            # 出图
            self.plot_patient_spearman(df_res, model_name, fold, fig_dir)

    # ---------------- 统一的日志记录与出图 ----------------
    def log_fold_summary(self, model_name, fold, t_metrics, v_metrics, summary_file=None):
        if summary_file is None:
            summary_file = os.path.join(self.model_dir, "5fold_metrics_summary.csv")
            
        file_exists = os.path.exists(summary_file)
        
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Model", "Fold_Or_Test", "Train_MAE", "Train_RMSE", "Train_R2", "Train_Pearson", "Train_Spearman", "Train_Kendall",
                                 "Val_MAE", "Val_RMSE", "Val_R2", "Val_Pearson", "Val_Spearman", "Val_Kendall"])
            if not t_metrics or not v_metrics:
                return
            writer.writerow([model_name, fold, 
                             f"{t_metrics['MAE']:.4f}", f"{t_metrics['RMSE']:.4f}", f"{t_metrics['R2']:.4f}", f"{t_metrics['Pearson']:.4f}", f"{t_metrics['Spearman']:.4f}", f"{t_metrics['Kendall']:.4f}",
                             f"{v_metrics['MAE']:.4f}", f"{v_metrics['RMSE']:.4f}", f"{v_metrics['R2']:.4f}", f"{v_metrics['Pearson']:.4f}", f"{v_metrics['Spearman']:.4f}", f"{v_metrics['Kendall']:.4f}"])

    def plot_patient_spearman(self, df_res, model_name, fold_name, fig_dir=None):
        if fig_dir is None:
            fig_dir = os.path.join(self.model_dir, 'fig', model_name, f'fold_{fold_name}')
        os.makedirs(fig_dir, exist_ok=True)
        
        actual_col = 'Normalized' if 'Normalized' in df_res.columns else df_res.columns[3]
        pred_col = 'Predicted'
        
        for sample in df_res['Sample'].unique():
            sub_df = df_res[df_res['Sample'] == sample].copy()
            if len(sub_df) < 2: 
                continue
                
            actual = sub_df[actual_col].values
            pred = sub_df[pred_col].values
            
            p_corr, _ = pearsonr(actual, pred)
            s_corr, _ = spearmanr(actual, pred)
            k_corr, _ = kendalltau(actual, pred)
            
            sub_df['Real_Rank'] = sub_df[actual_col].rank(method='min')
            sub_df['Pred_Rank'] = sub_df[pred_col].rank(method='min')
            sub_df = sub_df.sort_values(by='Real_Rank').reset_index(drop=True)
            
            plt.figure(figsize=(10, 6))
            plt.plot(sub_df.index, sub_df['Real_Rank'], label="Real Rank", marker='o', linestyle='-', color='royalblue')
            plt.plot(sub_df.index, sub_df['Pred_Rank'], label="Predicted Rank", marker='x', linestyle='--', color='darkorange')
            
            plt.title(f"[{model_name}] Patient: {sample}\nPearson: {p_corr:.4f} | Spearman: {s_corr:.4f} | Kendall: {k_corr:.4f}", fontsize=11)
            plt.xlabel("Compounds (Sorted by Real Efficacy Rank)")
            plt.ylabel("Rank (1 = Top Rank / Lowest Value)")
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{sample}_ranking_plot.png"), dpi=300)
            plt.close()

    # ---------------- 时间分割测试集集成评估 ----------------
    def run_ensemble_test(self, test_dir, test_out_dir):
        X_test, y_test, df_test = self._load_ml_data(test_dir)
        if len(X_test) == 0:
            self.logger.error("测试集加载失败，请检查文件路径！")
            return
            
        ensemble_summary_file = os.path.join(test_out_dir, "ensemble_test_metrics.csv")
        ml_models = ['RandomForest', 'SVM', 'XGBoost']
        
        for model_name in ml_models:
            self.logger.info(f"--- 评估测试集: {model_name} 集成 ---")
            fold_preds = []
            
            for fold in range(1, 6):
                # 加载对应 Fold 的模型权重
                model_path = os.path.join(self.model_dir, f'{model_name}_fold{fold}.pkl')
                model = joblib.load(model_path)
                
                # SVM 需要基于该 Fold 的特定 Scaler 对测试集进行归一化
                if model_name == 'SVM':
                    scaler_path = os.path.join(self.model_dir, f'ml_scaler_fold{fold}.pkl')
                    scaler = joblib.load(scaler_path)
                    X_input = scaler.transform(X_test)
                else:
                    X_input = X_test
                    
                # 记录该 Fold 在测试集上的预测结果
                preds = model.predict(X_input)
                fold_preds.append(preds)
                
            # 计算 5-fold 的平均预测值
            ensemble_pred = np.mean(fold_preds, axis=0)
            
            # 计算并记录指标
            test_metrics = self.calc_metrics(y_test, ensemble_pred)
            self.logger.info(f"[Ensemble {model_name}] Test MSE: {test_metrics['RMSE']**2:.4f} | Pearson: {test_metrics['Pearson']:.4f}")
            self.log_fold_summary(f"{model_name}_Ensemble", "Test", test_metrics, test_metrics, summary_file=ensemble_summary_file)
            
            # 保存预测结果与绘图
            df_res = df_test.copy()
            df_res['Predicted'] = ensemble_pred
            df_res.to_csv(os.path.join(test_out_dir, f"test_results_{model_name}_ensemble.csv"), index=False)
            
            fig_dir = os.path.join(test_out_dir, 'fig', f"{model_name}_Ensemble")
            self.plot_patient_spearman(df_res, f"{model_name}_Ensemble", "TestSet", fig_dir)

    # ---------------- 主运行流程 ----------------
    def run(self) -> None:
        base_dir = './2026_04_ds'
        test_dir = './2026_04_ds/time_test'
        test_out_dir = os.path.join(self.model_dir, 'time_test_ensemble_results')
        
        # 阶段一：5折交叉验证训练 (传统机器学习)
        self.logger.info("\n" + "*"*60 + "\n          PHASE 1: 5-Fold ML Training \n" + "*"*60)
        for fold in range(5):
            self.logger.info(f"\n{'='*50}\n          Starting Fold {fold + 1}/5\n{'='*50}")
            train_dir = os.path.join(base_dir, f'fold_{fold}', 'train')
            val_dir = os.path.join(base_dir, f'fold_{fold}', 'val')
            self.run_ml_baselines(train_dir, val_dir, fold + 1)
            
        self.logger.info("所有折数的训练已完成。")

        # 阶段二：测试集集成评估
        self.logger.info("\n" + "*"*60 + "\n          PHASE 2: Ensemble Testing \n" + "*"*60)
        os.makedirs(test_out_dir, exist_ok=True)
        self.run_ensemble_test(test_dir, test_out_dir)
        self.logger.info("测试集评估彻底完成！")


if __name__ == '__main__':
    CONFIG_PATH = "./config/exp.yml"
    
    pipeline = MLPipeline(root="./", yaml_path=CONFIG_PATH)
    pipeline.run()

