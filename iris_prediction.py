import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
import joblib

class IrisPredictor:
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 加载数据集
        print("正在加载数据集...")
        self.iris = load_iris()
        self.data = self.iris.data
        self.target = self.iris.target
        
        # 初始化模型
        print("\n初始化随机森林模型...")
        self.model = RandomForestClassifier(random_state=42)
        
        # 使用sklearn的train_test_split随机划分数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, 
            test_size=0.2,  # 保持80%训练，20%测试的比例
            random_state=42,  # 设置随机种子以保证结果可复现
            stratify=self.target  # 确保训练集和测试集中的类别比例一致
        )
        
        # 添加数据标准化
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def optimize_model(self):
        """模型参数优化"""
        print("\n开始模型参数优化...")
        
        # 减少参数搜索空间
        param_grid = {
            'n_estimators': [10, 50, 100],  # 减少树的数量选项
            'max_depth': [None, 10],        # 减少深度选项
            'min_samples_split': [2, 5],    # 减少分裂阈值选项
            'min_samples_leaf': [1, 2],     # 减少叶节点样本数选项
            'criterion': ['gini', 'entropy'] # 保留两种划分标准
        }
        
        # 使用网格搜索进行参数优化
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1  # 使用所有CPU核心加速计算
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print("\n最佳参数：")
        print(grid_search.best_params_)
        print(f"最佳得分：{grid_search.best_score_:.4f}")
        
        # 使用最佳参数更新模型
        self.model = grid_search.best_estimator_
    
    def cross_validate(self):
        """交叉验证"""
        print("\n执行5折交叉验证...")
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"交叉验证得分: {scores}")
        print(f"平均准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    def train_model(self):
        """训练模型"""
        print("\n开始训练模型...")
        print("使用前120个样本进行训练...")
        self.model.fit(self.X_train, self.y_train)
        print("模型训练完成！")
    
    def explore_data(self):
        """数据探索与可视化函数"""
        print("\n数据集基本信息：")
        print(f"样本数量: {len(self.data)}")
        print(f"特征数量: {self.data.shape[1]}")
        
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(self.data, columns=self.iris.feature_names)
        self.df['species'] = pd.Categorical.from_codes(self.target, self.iris.target_names)
        
        print("\n特征统计描述：")
        print(self.df.describe())
        
        # 调用各种可视化方法
        self._plot_feature_distributions()
        self._plot_feature_boxplots()
        self._plot_correlation_matrix()
        self._plot_pair_plot()
        self._plot_violin_plots()
    
    def _plot_feature_distributions(self):
        """绘制特征分布直方图"""
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(self.iris.feature_names, 1):
            plt.subplot(1, 4, i)
            for species in self.iris.target_names:
                data = self.df[self.df['species'] == species][feature]
                plt.hist(data, alpha=0.5, label=species, bins=15)
            plt.title(f'{feature} 分布')
            plt.xlabel('值')
            plt.ylabel('频数')
            if i == 1:
                plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_boxplots(self):
        """绘制箱线图"""
        plt.figure(figsize=(12, 6))
        for i, feature in enumerate(self.iris.feature_names, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(x='species', y=feature, data=self.df)
            plt.xticks(rotation=45)
            plt.title(f'{feature} 箱线图')
        plt.tight_layout()
        plt.show()
    
    def _plot_correlation_matrix(self):
        """绘制特征相关性热力图"""
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.drop('species', axis=1).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0)
        plt.title('特征相关性矩阵')
        plt.show()
    
    def _plot_pair_plot(self):
        """绘制特征对图"""
        sns.pairplot(self.df, hue='species', diag_kind='hist')
        plt.suptitle('特征对关系图', y=1.02)
        plt.show()
    
    def _plot_violin_plots(self):
        """绘制小提琴图"""
        plt.figure(figsize=(12, 6))
        for i, feature in enumerate(self.iris.feature_names, 1):
            plt.subplot(2, 2, i)
            sns.violinplot(x='species', y=feature, data=self.df)
            plt.xticks(rotation=45)
            plt.title(f'{feature} 分布（小提琴图）')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """绘制特征重要性"""
        importance = pd.DataFrame({
            'feature': self.iris.feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title('特征重要性排序')
        plt.xlabel('重要性得分')
        plt.show()
    
    def predict(self):
        """模型预测"""
        print("\n使用后30个样本进行预测...")
        predictions = self.model.predict(self.X_test)
        print("\n预测结果（前10个）：")
        print(predictions[:10])
        return predictions
    
    def evaluate_model(self):
        """评估模型"""
        predictions = self.predict()
        
        # 计算准确率
        accuracy = accuracy_score(self.y_test, predictions)
        print(f'\n准确率 Accuracy: {accuracy:.4f}')
        
        # 输出详细的评估报告
        print("\n详细评估报告：")
        print("TP: 将正类预测为正类数（预测正确）")
        print("FN: 将正类预测为负类数（预测错误）")
        print("FP: 将负类预测为正类数（预测错误）")
        print("TN: 将负类预测为负类数（预测正确）")
        print("\n分类报告：")
        
        # 获取实际出现的类别
        unique_labels = np.unique(np.concatenate([self.y_test, predictions]))
        target_names = [self.iris.target_names[i] for i in unique_labels]
        
        print(classification_report(self.y_test, predictions,
                                  target_names=target_names))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.show()
    
    def predict_proba(self):
        """输出预测概率"""
        print("\n预测概率...")
        probabilities = self.model.predict_proba(self.X_test)
        
        for i, prob in enumerate(probabilities[:10]):  # 显示前10个样本的概率
            print(f"样本 {i+1} 的预测概率:")
            for class_name, p in zip(self.iris.target_names, prob):
                print(f"{class_name}: {p:.4f}")
            print() 
    
    def plot_learning_curve(self):
        """绘制学习曲线"""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='训练集得分')
        plt.plot(train_sizes, test_mean, label='验证集得分')
        plt.fill_between(train_sizes, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, 
                        test_mean + test_std, alpha=0.1)
        plt.xlabel('训练样本数')
        plt.ylabel('得分')
        plt.title('学习曲线')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def select_features(self):
        """特征选择"""
        selector = SelectFromModel(self.model, prefit=True)
        feature_idx = selector.get_support()
        selected_features = [f for f, s in zip(self.iris.feature_names, feature_idx) if s]
        
        print("\n选择的特征:")
        for feature in selected_features:
            print(f"- {feature}")
    
    def save_model(self, filename='iris_model.joblib'):
        """保存模型"""
        joblib.dump(self.model, filename)
        print(f"\n模型已保存到: {filename}")
    
    def load_model(self, filename='iris_model.joblib'):
        """加载模型"""
        self.model = joblib.load(filename)
        print(f"\n已加载模型: {filename}")
    
    def export_results(self, filename='predictions.csv'):
        """导出预测结果"""
        predictions = self.predict()
        probabilities = self.model.predict_proba(self.X_test)
        
        results_df = pd.DataFrame({
            'True_Label': [self.iris.target_names[i] for i in self.y_test],
            'Predicted_Label': [self.iris.target_names[i] for i in predictions]
        })
        
        # 添加预测概率
        for i, class_name in enumerate(self.iris.target_names):
            results_df[f'Prob_{class_name}'] = probabilities[:, i]
            
        results_df.to_csv(filename, index=False)
        print(f"\n预测结果已保存到: {filename}")