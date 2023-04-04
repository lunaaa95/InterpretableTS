from json.tool import main
import mailbox
from operator import imod
from tkinter import Y
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
import argparse
import pathlib

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default='100',
                    help='num of max epochs')
parser.add_argument('--device', type=str, default='cpu',
                    help='device')
parser.add_argument('--teacher-base-dir', type=str, default='../GRU',
                    help='base dir of teacher, default: GRU')



def load_data(args, DEVICE):    
    data_base_url = pathlib.Path(args.teacher_base_dir) / 'data'
    label_base_url = pathlib.Path(args.teacher_base_dir) / 'preds'

    # 1. 读 data 文件
    train_data = read_pickle(data_base_url / f'train.pkl')
    valid_data = read_pickle(data_base_url / f'valid.pkl')
    test_data = read_pickle(data_base_url / f'test.pkl')
    
    # 原数据是 list，转成 torch
    train_data = torch.tensor(train_data, device=DEVICE)
    valid_data = torch.tensor(valid_data, device=DEVICE)
    test_data = torch.tensor(test_data, device=DEVICE)
    train_data = train_data.to(torch.double)
    valid_data = valid_data.to(torch.double)
    test_data = test_data.to(torch.double)


    print(train_data.shape, valid_data.shape, test_data.shape)
    # (119624, 30, 4) (14953, 30, 4) (14959, 30, 4)
    
    # 2. 读 label 文件，为 GRU 训出来的 label
    train_label = torch.from_numpy(read_pickle(label_base_url / f'train_pred.pkl')).to(DEVICE)
    valid_label = torch.from_numpy(read_pickle(label_base_url / f'valid_pred.pkl')).to(DEVICE)
    test_label = torch.from_numpy(read_pickle(label_base_url / f'test_pred.pkl')).to(DEVICE)
    train_label = train_label.to(torch.double)
    valid_label = valid_label.to(torch.double)
    test_label = test_label.to(torch.double)

    print(train_label.shape, valid_label.shape, test_label.shape)
    # (119624, 1) (14953, 1) (14959, 1)
    
    return train_data, valid_data, test_data, train_label, valid_label, test_label


def train(model, data, label):
    # feature 维度拉平
    data = data.reshape((data.shape[0], -1))
    # print(data.shape, label.shape)
    # torch.Size([119624, 120]) torch.Size([119624, 1])
    # 拟合
    model.fit(data, label)
    pass


def evaluate(model, data, label, metrics):
    data = data.reshape((data.shape[0], -1))
    y_pred = model.predict(data)
    
    y_pred = y_pred.reshape(-1, 1)
    label = label.reshape(-1, 1)
    fidelity, auc, acc = metrics(y_pred=y_pred, y_true=label)
    return fidelity, auc, acc


def main():
    args = parser.parse_args()
    device = args.device
    
    # 1. 加载数据集
    print('================== load data ======================')
    train_data, valid_data, test_data, train_label, valid_label, test_label = load_data(args, device)
    
    # 2. 训练
    # x：(30, 4) 即 120 个特征
    # Y：(1, 1) 即第 31 天的收盘价
    model = Lasso(alpha=0.5)
    # 在训练集上拟合
    train(model=model, data=train_data, label=train_label)
    
    # 3. 评估
    eval_metrics = metrics_classify
    print('===================== evaluate valid data =======================')
    valid_fidelity, valid_auc, valid_acc = evaluate(model, valid_data, valid_label, eval_metrics)
    print('===================== evaluate test data ========================')
    test_fidelity, test_auc, test_acc = evaluate(model, test_data, test_label, eval_metrics)

    print('=============== valid res ===============')
    print_str = 'valid fidelity: {:.4f}, valid auc: {:.4f}, valid acc: {:.4f}'.format(valid_fidelity, valid_auc, valid_acc)
    print(print_str)
    
    
    print('=============== test res ===============')
    print_str = 'test fidelity: {:.4f}, test auc: {:.4f}, test acc: {:.4f}'.format(test_fidelity, test_auc, test_acc)
    print(print_str)
    
    pass



if __name__ == "__main__":
    main()
