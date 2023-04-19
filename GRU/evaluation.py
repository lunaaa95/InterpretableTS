'''
    根据 gru_rnn.py 生成的预测结果文件和 groundtruth 计算 acc, auc 等相关指标
'''
import pathlib
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

# pickle 文件读写
def read_pickle(url):
    with open(url, 'rb') as f:
        res = pickle.load(f)
    return res


pred_base_url = pathlib.Path('preds')
ground_truth_base_url = pathlib.Path('../data/stock100_teacher')

train_pred = np.array(read_pickle('preds/train_pred.pkl')).squeeze()
valid_pred = np.array(read_pickle('preds/valid_pred.pkl')).squeeze()
test_pred = np.array(read_pickle('preds/test_pred.pkl')).squeeze()

train_label = np.array(read_pickle('../data/stock100_teacher/train_label.pkl')).squeeze()
valid_label = np.array(read_pickle('../data/stock100_teacher/valid_label.pkl')).squeeze()
test_label = np.array(read_pickle('../data/stock100_teacher/test_label.pkl')).squeeze()

print(train_label.shape, train_pred.shape)
print(train_label[0] == train_pred[0])

def eval(y_pred, y_true):
    acc = (sum(y_pred == y_true) / len(y_true)).item()
    auc = roc_auc_score(y_true, y_pred)
    return acc, auc

# train_acc, train_auc = eval(train_pred, train_label)
# print('=============== train res ===============')
# print_str = 'train auc: {:.4f}, train acc: {:.4f}'.format(train_auc, train_acc)
# print(print_str)

# valid_acc, valid_auc = eval(valid_pred, valid_label)
# print('=============== train res ===============')
# print_str = 'valid auc: {:.4f}, valid acc: {:.4f}'.format(valid_auc, valid_acc)
# print(print_str)

# test_acc, test_auc = eval(test_pred, test_label)
# print('=============== train res ===============')
# print_str = 'test auc: {:.4f}, test acc: {:.4f}'.format(test_auc, test_acc)
# print(print_str)
