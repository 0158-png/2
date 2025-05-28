import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import joblib
import logging
from datetime import datetime, UTC
import os
from train import DeepNN  # 更新为新的模型名称

# 配置日志
logging.basicConfig(level=logging.INFO)


def load_models(nn_path='checkpoints/best_nn_model.pth',
                gb_path='checkpoints/best_gb_model.pkl'):
    """加载训练好的模型"""
    device = torch.device('cpu')

    # 加载神经网络模型
    nn_model = DeepNN().to(device)
    nn_model.load_state_dict(torch.load(nn_path))
    nn_model.eval()

    # 加载梯度提升模型
    gb_model = joblib.load(gb_path)

    return nn_model, gb_model


def ensemble_predict(nn_model, gb_model, features, device):
    """使用集成模型进行预测"""
    # 神经网络预测
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(device)
        nn_pred = nn_model(features_tensor).argmax(1).cpu().numpy()

    # 梯度提升预测
    gb_pred = gb_model.predict(features)

    # 集成预测（多数投票）
    ensemble_pred = np.array([nn_pred, gb_pred])
    final_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=ensemble_pred
    )

    return final_pred


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()


def main():
    try:
        # 加载测试数据
        pos_data = pd.read_csv('output/data/positive_features.csv')
        neg_data = pd.read_csv('output/data/negative_features.csv')

        all_data = pd.concat([pos_data, neg_data], axis=0)
        features = all_data.iloc[:, 2:].values
        labels = all_data.iloc[:, 1].values

        # 标准化特征
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # 加载模型
        device = torch.device('cpu')
        nn_model, gb_model = load_models()

        # 进行预测
        predictions = ensemble_predict(nn_model, gb_model, features, device)

        # 计算准确率
        accuracy = accuracy_score(labels, predictions) * 100

        # 生成分类报告
        class_report = classification_report(labels, predictions)

        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)

        # 打印结果
        logging.info("=" * 50)
        logging.info("评估结果:")
        logging.info("=" * 50)
        logging.info(f"整体准确率: {accuracy:.2f}%")
        logging.info("\n分类报告:")
        logging.info(class_report)

        # 绘制混淆矩阵
        plot_confusion_matrix(cm)

        # 保存预测结果
        results_df = pd.DataFrame({
            'True_Label': labels,
            'Predicted_Label': predictions
        })
        results_df.to_csv('evaluation_results.csv', index=False)

        # 保存评估指标
        evaluation_metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'evaluation_time': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'evaluated_by': 'syy3308'
        }

        with open('evaluation_metrics.json', 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)

        logging.info("\n评估结果已保存到 evaluation_results.csv 和 evaluation_metrics.json")
        logging.info("混淆矩阵可视化已保存到 confusion_matrix.png")

    except Exception as e:
        logging.error(f"评估过程出错: {e}")
        raise


if __name__ == "__main__":
    current_time = datetime.now(UTC)
    logging.info(f"开始评估时间: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    try:
        main()
    except Exception as e:
        logging.error(f"程序执行出错: {e}")