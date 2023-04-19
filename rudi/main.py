from json.tool import main
from math import ceil
from operator import imod
from statistics import mode
from tkinter import Y
from tokenize import Double
import numpy as np
import pandas as pd
import torch
import argparse
import pathlib
from torch import optim
import os
from tqdm import tqdm
from sklearn import preprocessing
import random

random.seed(1223)
torch.manual_seed(1223)

from mlp import MLP
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default='50',
                    help='num of max epochs')
parser.add_argument('--batch-size', type=int, default='16',
                    help='batch size')
parser.add_argument('--lr', type=float, default='0.0001',
                    help='learning rate')
parser.add_argument('--clip', type=float, default='0.2',
                    help='clip')
parser.add_argument('--device', type=str, default='cpu',
                    help='device')
parser.add_argument('--data-base-dir', type=str, default='../rudi',
                    help='base dir of data, default: ../rudi')
parser.add_argument('--teacher-base-dir', type=str, default='../GRU/preds',
                    help='base dir of teacher, default: ../GRU/preds')
parser.add_argument('--true-base-dir', type=str, default='../data/stock100_teacher',
                    help='base dir of ground truth, default: ../data/stock100_teacher')



def load_data(args, DEVICE):    
    data_base_url = pathlib.Path(args.data_base_dir)
    label_base_url = pathlib.Path(args.teacher_base_dir)
    ground_truth_base_url = pathlib.Path(args.true_base_dir)

    # 1. 读 data 文件
    train_data = read_pickle(data_base_url / f'train_features.pkl')
    valid_data = read_pickle(data_base_url / f'valid_features.pkl')
    test_data = read_pickle(data_base_url / f'test_features.pkl')
    
    # 正则化
    train_data = preprocessing.scale(train_data)
    valid_data = preprocessing.scale(valid_data)
    test_data = preprocessing.scale(test_data)
    
    # 原数据是 numpy，转成 torch
    train_data = torch.from_numpy(train_data).to(DEVICE)
    valid_data = torch.from_numpy(valid_data).to(DEVICE)
    test_data = torch.from_numpy(test_data).to(DEVICE)
    train_data = train_data.to(torch.float)
    valid_data = valid_data.to(torch.float)
    test_data = test_data.to(torch.float)

    print(train_data.shape, valid_data.shape, test_data.shape)
    # (119624, 40) (14953, 40) (14959, 40)
    
    # 2. 读 teacher 训出来的 label
    train_teacher_label = torch.from_numpy(read_pickle(label_base_url / f'train_pred.pkl')).to(DEVICE)
    valid_teacher_label = torch.from_numpy(read_pickle(label_base_url / f'valid_pred.pkl')).to(DEVICE)
    test_teacher_label = torch.from_numpy(read_pickle(label_base_url / f'test_pred.pkl')).to(DEVICE)
    train_teacher_label = train_teacher_label.to(torch.double)
    valid_teacher_label = valid_teacher_label.to(torch.double)
    test_teacher_label = test_teacher_label.to(torch.double)

    print(train_teacher_label.shape, valid_teacher_label.shape, test_teacher_label.shape)
    # (119624, 1) (14953, 1) (14959, 1)
    
    # 3. 读 ground truth label
    train_label = torch.from_numpy(np.array(read_pickle(ground_truth_base_url / f'train_label.pkl'))).to(DEVICE)
    valid_label = torch.from_numpy(np.array(read_pickle(ground_truth_base_url / f'valid_label.pkl'))).to(DEVICE)
    test_label = torch.from_numpy(np.array(read_pickle(ground_truth_base_url / f'test_label.pkl'))).to(DEVICE)
    train_label = train_label.to(torch.double)
    valid_label = valid_label.to(torch.double)
    test_label = test_label.to(torch.double)
    
    print(train_label.shape, valid_label.shape, test_label.shape)
    # (119624, 1) (14953, 1) (14959, 1)
    
    return train_data, valid_data, test_data, train_teacher_label, valid_teacher_label, test_teacher_label, train_label, valid_label, test_label


def train(model, data, label, args, optimizer):
    model.train()
    batch_size = args.batch_size
    total_batch_count = list(range(ceil(len(data) / batch_size)))
    random.shuffle(total_batch_count)
    total_loss = 0.0
    total_loss_count = 0
    
    def loss_criterion_1(pred, y_train):
        loss = (-y_train * torch.log(pred + 1e-8) - (1 - y_train) * torch.log(1 - pred + 1e-8)).mean()
        return loss
    loss_helper = loss_criterion_1
    
    for i in total_batch_count:
        data_batch = data[i * batch_size: (i + 1) * batch_size]
        label_batch = label[i * batch_size: (i + 1) * batch_size]
        output = model(data_batch)
        loss = loss_helper(output.squeeze(), label_batch.squeeze())
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_size == batch_size - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_size != batch_size - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return total_loss / total_loss_count


def evaluate(model, data, teacher_label, true_label, metrics, args, type=0):
    model.eval()
    batch_size = args.batch_size
    total_batch_count = ceil(len(data) / batch_size)
    preds = []
    
    for i in range(total_batch_count):
        data_batch = data[i * batch_size: (i + 1) * batch_size]
        output = model(data_batch).detach().cpu().numpy()
        preds.extend(output)
    
    if type:
        fidelity = metrics(y_pred=preds, y_true=teacher_label)
        acc, auc = eval(preds, true_label)
    else:
        fidelity = 0
        acc, auc = eval(preds, teacher_label)

    return fidelity, auc, acc


def main():
    args = parser.parse_args()
    device = args.device
    epochs = args.epochs
    
    # 1. 加载数据集
    print('================== load data ======================')
    train_data, valid_data, test_data, train_teacher_label, valid_teacher_label, test_teacher_label, train_label, valid_label, test_label = load_data(args, device)
    
    len_features = train_data.shape[1]
    
    # 2. 训练
    model = MLP(len_features, 32, 1)
    optimizer = optim.Adam(model.parameters(), lr= args.lr)
    
    best_valid_auc = -1
    best_model_file = None
    eval_metrics = metrics_classify
    
    for epoch in tqdm(range(epochs)):
        train_loss = train(model=model, data=train_data, label=train_teacher_label, args=args, optimizer=optimizer)
        _, valid_auc, valid_acc = evaluate(model, valid_data, valid_teacher_label, valid_label, eval_metrics, args)
        _, test_auc, test_acc = evaluate(model, test_data, test_teacher_label, test_label, eval_metrics, args)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            eval_best_str = "epoch{}, train_loss{:.4f}, valid_auc{:.4f}, valid_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}".format(
                epoch, train_loss, valid_auc, valid_acc, test_auc, test_acc
            )
            wait_epoch = 0
            if best_model_file:
                os.remove(best_model_file)
            best_model_file = "./SavedModels/valid#auc{}_acc{}_test#auc{}_acc{}".format(valid_auc, valid_acc, test_auc, test_acc)
            torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch > 50:
            print("saved_model_result:",eval_best_str)
            break
        epoch += 1
        eval_str = "epoch{}, train_loss{:.4f}, valid_auc{:.4f}, valid_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}".format(
                epoch, train_loss, valid_auc, valid_acc, test_auc, test_acc
            )
        print(eval_str)
    
    
    # 读取最好的模型去算几个指标
    print('-------------- train finish ! ---------------')
    model.load_state_dict(torch.load(best_model_file))
    valid_fidelity, valid_auc, valid_acc = evaluate(model, valid_data, valid_teacher_label, valid_label, eval_metrics, args, type=1)
    test_fidelity, test_auc, test_acc = evaluate(model, test_data, test_teacher_label, test_label, eval_metrics, args, type=1)
    eval_str = "epoch{}, train_loss{:.4f}, valid_fidelity{:.4f}, valid_auc{:.4f}, valid_acc{:.4f}, test_fidelity{:.4f}, test_auc{:.4f}, test_acc{:.4f}".format(
                epoch, train_loss, valid_fidelity, valid_auc, valid_acc, test_fidelity, test_auc, test_acc
            )
    print('------------------ final result ----------------')
    print(eval_str)
    
    pass



if __name__ == "__main__":
    main()

