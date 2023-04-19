import pickle
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def read_pickle(url):
    with open(url, 'rb') as f:
        res = pickle.load(f)
    return res


def to_pickle(url, data):
    with open(url, 'wb') as f:
        pickle.dump(data, f)

def metrics_reg(y_pred, y_true):
    """
    回归问题的评估函数

    Args:
        y_pred (_type_): 模型的预测值
        y_true (_type_): 真实值即 label，这里为 teacher 训原数据得到的 label
    """
    print(y_pred.shape, y_true.shape)
    # fidelity
    
    # auc
    return 0, 0

def metrics_classify(y_pred, y_true):
    """
    分类问题的评估函数

    Args:
        y_pred (_type_): 模型的预测值
        y_true (_type_): 真实值即 label，这里为 teacher 训原数据得到的 label
    """
    # 预测值转为 [0, 1]
    y_pred = np.round(y_pred)
    y_true = np.array(y_true)
    
    # fidelity
    sz = len(y_pred)
    # sz = 100
    # y_pred = y_pred[:100]
    # y_true = y_true[:100]
    true_count = 0
    for i in tqdm(range(sz)):
        for j in range(sz):
            if i == j:
                continue
            x1 = y_pred[i] > y_pred[j]
            x2 = y_true[i] > y_true[j]
            if x1 == x2:
                true_count += 1
    fidelity = true_count / (sz * (sz - 1))
    
    return fidelity

def eval(y_pred, y_true):
    y_pred = np.round(np.array(y_pred))
    y_true = np.array(y_true).reshape((-1, 1))
    acc = (sum(y_pred == y_true) / len(y_true)).item()
    auc = roc_auc_score(y_true, y_pred)
    return acc, auc
