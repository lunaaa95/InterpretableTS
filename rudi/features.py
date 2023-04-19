'''
    根据 rules 构建特征
'''
import torch
import numpy as np
import pickle
import pathlib
import pandas as pd


def get_features(data):
    # data: (N, 20). 20 为 stats 个数
    sz = len(data)
    features_all = np.zeros((sz, LEN_RULES))
    # 40 个系数
    rule_values = [1.3468e+04, 1.3456e+04, 1.3173e+04, 1.3154e+04, 1.2627e+04, 1.2256e+04, 1.1851e+04, 1.1150e+04, 1.1005e+04, 1.0364e+04, 9.9317e+03, 8.5132e+03, 8.4746e+03, 8.3372e+03,
                   8.1769e+03, 5.1787e+03, 3.5977e+03, -2.8140e+00, -2.7314e+00, 2.0905e+00, 1.6156e+00, 1.4076e+00, -1.3251e+00, -1.0982e+00, -1.0160e+00, 1.0096e+00, 7.8697e-01, 
                   -7.2691e-01, 6.4372e-01, 6.4057e-01, -5.1289e-01, -4.9347e-01, -4.8394e-01, -4.4636e-01, 3.9475e-01, -3.2856e-01, -3.0109e-01, -1.6871e-01, 1.5927e-01, 1.2244e-01]
    # rules 和 stats 对应关系
    features_all[:, 0] = (data[:, 7] > 1.0610e+01) | (np.logical_not(data[:, 15] > 1.0139e+02))
    features_all[:, 1] = ((data[:, 17] > 1.7151e+01) | (data[:, 6] > 8.2000e-01)) | (np.logical_not(data[:, 8] > 1.0364e+02))
    features_all[:, 2] = (data[:, 3] > 1.0579e+01) | (np.logical_not(data[:, 11] > 1.0175e+02))
    features_all[:, 3] = (np.logical_not(data[:, 2] > 1.0450e+02) | (np.logical_not(data[:, 15] > 1.0139e+02)))
    features_all[:, 4] = (data[:, 0] > 8.2000e-01) | (np.logical_not(data[:, 3] > 1.0567e+02))
    features_all[:, 5] = (np.logical_not(data[:, 7] > 4.7800e+02)) | (
        (data[:, 11] > 8.4750e+00) | (np.logical_not(data[:, 0] > 7.5000e+00)) | 
        (np.logical_not(data[:, 6] > 1.0711e+02))
    )
    features_all[:, 6] = (np.logical_not(data[:, 12] > 6.0575e+00)) | (np.logical_not(data[:, 16] > 1.0204e+02))
    features_all[:, 7] = ((np.logical_not(data[:, 3] > 2.7205e+01)) | (np.logical_not(data[:, 2] > 6.3995e+00))) | (data[:, 17] > 7.0850e-01)
    features_all[:, 8] = (np.logical_not(data[:, 3] > 1.0567e+02)) | (
        (np.logical_not(data[:, 14] > 1.0712e+02)) | (data[:, 16] > 8.2000e-01)
    )
    features_all[:, 9] = (data[:, 11] > 3.9000e-01) | (np.logical_not(data[:, 18] > 9.4850e+01))
    
    features_all[:, 10] = (data[:, 11] > 3.9000e-01) | (np.logical_not(data[:, 10] > 1.0204e+02))
    features_all[:, 11] = ((data[:, 13] > 8.2000e-01) | (np.logical_not(data[:, 7] > 2.9878e+03))) | (np.logical_not(data[:, 9] > 1.0712e+02))
    features_all[:, 12] = (np.logical_not(data[:, 11] > 1.0175e+02)) | (data[:, 13] > 8.2000e-01)
    features_all[:, 13] = (np.logical_not(data[:, 6] > 1.0711e+02)) | (np.logical_not(data[:, 4] > 1.0712e+02))
    features_all[:, 14] = (data[:, 18] > 2.4000e-01) | (np.logical_not(data[:, 19] > 1.1050e+01))
    features_all[:, 15] = (np.logical_not(data[:, 9] > 1.0712e+02)) | (np.logical_not(data[:, 2] > 1.0450e+02))
    features_all[:, 16] = (np.logical_not(data[:, 9] > 2.7250e+01)) | (data[:, 1] > 8.2000e-01)
    features_all[:, 17] = (data[:, 7] > 7.4425e+02) & (
        (data[:, 2] > 1.0450e+02) & (data[:, 15] > 1.0292e+01)
    )
    features_all[:, 18] = (data[:, 13] > 1.0204e+02) & (np.logical_not(data[:, 6] > 8.8500e+00))
    features_all[:, 19] = (np.logical_not(data[:, 5] > 6.5255e+00)) & (data[:, 11] > 1.0175e+02)
    
    features_all[:, 20] = (data[:, 1] > 1.0711e+02) & (data[:, 17] > 1.0455e+02)
    features_all[:, 21] = (data[:, 1] > 1.0711e+02) & (data[:, 5] > 2.7581e+01)
    features_all[:, 22] = (np.logical_not(data[:, 1] > 4.2500e+00)) & (data[:, 18] > 2.2450e+01)
    features_all[:, 23] = (np.logical_not(data[:, 10] > 8.2000e-01)) & (data[:, 0] > 1.0204e+02)
    features_all[:, 24] = ((np.logical_not(data[:, 12] > 4.0275e+00)) & (np.logical_not(data[:, 0] > 4.1700e+00))) & (
        (data[:, 10] > 1.0204e+02) & (data[:, 16] > 1.0204e+02)
    )
    features_all[:, 25] = (data[:, 11] > 1.0175e+02) & (np.logical_not(data[:, 4] > 8.2000e-01))
    features_all[:, 26] = (np.logical_not(data[:, 19] > 1.4640e+01)) | (np.logical_not(data[:, 18] > 1.4500e+01))
    features_all[:, 27] = (np.logical_not(data[:, 2] > 7.0850e-01)) & (data[:, 7] > 2.9878e+03)
    features_all[:, 28] = (data[:, 12] > 1.6203e+01) & (np.logical_not(data[:, 7] > 1.0610e+01))
    features_all[:, 29] = (np.logical_not(data[:, 18] > 2.2450e+01)) | (np.logical_not(data[:, 9] > 1.3020e+01))
    
    features_all[:, 30] = (data[:, 3] > 1.0579e+01) | (data[:, 6] > 1.0640e+01)
    features_all[:, 31] = (data[:, 3] > 1.3082e+01) & (data[:, 6] > 1.7250e+01)
    features_all[:, 32] = (np.logical_not(data[:, 1] > 4.2500e+00)) & (data[:, 18] > 9.4850e+01)
    features_all[:, 33] = (data[:, 6] > 1.3140e+01) & (data[:, 2] > 1.0507e+01)
    features_all[:, 34] = (np.logical_not(data[:, 0] > 8.2000e-01)) & (data[:, 5] > 1.0662e+02)
    features_all[:, 35] = (data[:, 11] > 1.2488e+01) & (np.logical_not(data[:, 16] > 7.4700e+00))
    features_all[:, 36] = (np.logical_not(data[:, 18] > 2.4000e-01)) & (data[:, 5] > 1.0662e+02)
    features_all[:, 37] = (np.logical_not(data[:, 6] > 8.2000e-01)) & (data[:, 14] > 1.0712e+02)
    features_all[:, 38] = (data[:, 10] > 1.0204e+02) & (np.logical_not(data[:, 4] > 8.2000e-01))
    features_all[:, 39] = (np.logical_not(data[:, 4] > 8.2000e-01)) & (data[:, 6] > 2.7330e+01)
    
    features_all = features_all * rule_values
    
    return features_all


def to_pickle(url, data):
    with open(url, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    LEN_RULES = 40
    url = pathlib.Path('../data/stock100/gru_stats')
    
    train_stats = np.array(pd.read_csv(url / f'train_stats.csv'))
    valid_stats = np.array(pd.read_csv(url / f'valid_stats.csv'))
    test_stats = np.array(pd.read_csv(url / f'test_stats.csv'))
    
    train_stats = train_stats[:, 1:]
    valid_stats = valid_stats[:, 1:]
    test_stats = test_stats[:, 1:]
    
    print('---------------- load data success -----------------')
    
    train_features = get_features(train_stats)
    to_pickle('train_features.pkl', train_features)
    print('---------------- get train data features success -----------------')
    
    valid_features = get_features(valid_stats)
    to_pickle('valid_features.pkl', valid_features)
    print('---------------- get valid data features success -----------------')
    
    test_features = get_features(test_stats)
    to_pickle('test_features.pkl', test_features)
    print('----------------get test data features success -----------------')
