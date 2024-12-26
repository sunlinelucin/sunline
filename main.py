from iris_prediction import IrisPredictor

def main():
    # 创建预测器实例
    predictor = IrisPredictor()
    
    # 数据探索
    predictor.explore_data()
    
    # 模型优化
    predictor.optimize_model()
    
    # 交叉验证
    predictor.cross_validate()
    
    # 训练模型
    predictor.train_model()
    
    # 特征选择
    predictor.select_features()
    
    # 绘制学习曲线
    predictor.plot_learning_curve()
    
    # 预测概率
    predictor.predict_proba()
    
    # 评估模型
    predictor.evaluate_model()
    
    # 保存模型和结果
    predictor.save_model()
    predictor.export_results()

if __name__ == "__main__":
    main()
