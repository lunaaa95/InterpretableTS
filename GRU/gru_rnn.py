import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import math
from tqdm import tqdm
import os
import random

random.seed(1223)
torch.manual_seed(1223)


epoch = 300
feature_num = 4
batch_size = 32

class GRU_model(nn.Module):
    def __init__(self, input_size=feature_num, hid_size=16):
        super(GRU_model, self).__init__()
        self.gru_layer = nn.GRU(input_size=input_size, hidden_size=hid_size, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hid_size, 1), nn.Sigmoid())
    
    def forward(self, x):
        # x: tensor(batch, timestamps, feats) : (32, 30, 4)
        _, hid = self.gru_layer(x)
        hid = hid.squeeze(0) # (32, 64)
        pred = self.linear(hid)  # (32, 1)
        return pred

def train(model, x_train, y_train):
    model.train()
    # cross entropy loss
    def loss_criterion_1(pred, y_train):
        loss = (-y_train * torch.log(pred + 1e-8) - (1 - y_train) * torch.log(1 - pred + 1e-8)).mean()
        return loss
    
    indexs = list(range(math.ceil(x_train.shape[0] / batch_size)))  # 总共要训的 batch 轮数
    random.shuffle(indexs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_criterion = loss_criterion_1
    total_loss = 0
    total_loss_count = 0
    log_inteval = batch_size  # 打印 loss 信息的 batch 间隔
    
    # 训练
    for index in tqdm(indexs):
        x = x_train[index * batch_size: (index + 1) * batch_size] # (32, 30, 4)
        y = y_train[index * batch_size: (index + 1) * batch_size].reshape(-1, 1) # (32, 1)
        pred = model(x)
        loss = loss_criterion(pred, y)
        loss.backward()
        total_loss += loss
        total_loss_count += 1
        if total_loss_count % log_inteval == log_inteval - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % log_inteval != log_inteval - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return total_loss / len(indexs)  

def evaluate_1(model, x_eval, y_eval, save=False, url=None):
    model.eval()
    # indexs = list(range(x_eval.shape[0]))
    indexs = list(range(math.ceil(x_eval.shape[0] / batch_size)))  # 总共要训的 batch 轮数
    preds = 0
    trues = 0
    outputs = []
    true_outputs = []
    for index in tqdm(indexs):
        x = x_eval[index * batch_size: (index + 1) * batch_size] # (32, 30, 4)
        y = y_eval[index * batch_size: (index + 1) * batch_size].reshape(-1, 1) # (32, 1)
        true_outputs.append(y)
        output = model(x) # (32, 1)
        temp0 = torch.zeros_like(y)
        temp1 = torch.ones_like(y)
        output = torch.where(output > 0.5, temp1, temp0)
        outputs.append(output)
        true_num = temp1[output == y].sum()
        preds += x.shape[0]
        trues +=true_num
    outputs = torch.cat(outputs, dim=0)
    if save:
        to_pickle(url, outputs.cpu().numpy())
    true_outputs = torch.cat(true_outputs, dim=0)
    auc = roc_auc_score(outputs.cpu().numpy(), true_outputs.cpu().numpy())
    return auc, trues/preds


# pickle 文件读写
def read_pickle(url):
    with open(url, 'rb') as f:
        res = pickle.load(f)
    return res


def to_pickle(url, data):
    with open(url, 'wb') as f:
        pickle.dump(data, f)

    

if __name__ == "__main__":
    DEVICE = 'cpu'
    # if torch.cuda.is_available:
    #     DEVICE = "cuda:0"\
    print('load data')
    # 读取数据
    train_x = torch.from_numpy(np.array(read_pickle('data/train.pkl')).astype('float'))
    train_y = torch.from_numpy(np.array(read_pickle('data/train_label.pkl')).astype('float'))
    eval_x = torch.from_numpy(np.array(read_pickle('data/valid.pkl')).astype('float'))
    eval_y = torch.from_numpy(np.array(read_pickle('data/valid_label.pkl')).astype('float'))
    test_x = torch.from_numpy(np.array(read_pickle('data/test.pkl')).astype('float'))
    test_y = torch.from_numpy(np.array(read_pickle('data/test_label.pkl')).astype('float'))

    print(train_x.shape, train_y.shape)
    
    model = GRU_model()
    model = model.to(device=DEVICE)
    model = model.to(torch.double)
    evaluate = evaluate_1
    task = 1
    clip = 0.25
    
    if task:
        best_acc = 0
        best_eval_auc = 0
        best_epoch = 0
        wait_epoch = 0
        best_epoch_test_acc = 0
        best_model_file = None
        
        for i in range(epoch):
            train_loss = train(model, train_x, train_y)
            train_auc, train_acc = evaluate(model, train_x, train_y)
            eval_auc, eval_acc = evaluate(model, eval_x, eval_y)
            test_auc, test_acc = evaluate(model, test_x, test_y)
            # train_auc, train_acc = evaluate(model, train_x, train_y, True, './preds/train_preds.pkl')
            # eval_auc, eval_acc = evaluate(model, eval_x, eval_y, True, './preds/valid_preds.pkl')
            # test_auc, test_acc = evaluate(model, test_x, test_y, True, './preds/test_preds.pkl')
            # exit
            
            if eval_auc > best_eval_auc:
                eval_epoch_best = eval_auc
                eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(i, train_loss, eval_auc,eval_acc, test_auc, test_acc)
                wait_epoch = 0
                best_epoch = i
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./SavedModels/train#auc{}_acc{}_eval#auc{}_acc{}_test#auc{}_acc{}".format(train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc)
                torch.save(model.state_dict(), best_model_file)
            else:
                wait_epoch += 1

            if wait_epoch > 50:
                print("saved_model_result:",eval_best_str)
                break
            i += 1
            eval_str = "epoch{}, train_loss{:.4f}, train_auc{:.4f}, train_acc{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(i, train_loss, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc)
            
            print(eval_str)

        print("the best epoch is {0}: best_eval_auc: {1}, best_eval_acc: {2}, test_acc: {3}".format(best_epoch , best_eval_auc, best_acc, best_epoch_test_acc))
        
        # 导出文件
        train_auc, train_acc = evaluate(model, train_x, train_y, True, './preds/train')
        eval_auc, eval_acc = evaluate(model, eval_x, eval_y, True, './preds/valid')
        test_auc, test_acc = evaluate(model, test_x, test_y, True, './preds/test')

