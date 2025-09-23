import json
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_dict(file_path):
    with open(file_path, 'r') as f:
        data_dict = json.load(f)

    return data_dict


def load_data(file_path):
    return pd.read_csv(file_path)


def mz_to_vec(mz):
    mz = int(round(mz, 2) / 0.01)
    vec = np.zeros(50000, dtype=np.float32)
    vec[mz] = 1

    return vec


def spec_to_vec(spec):
    valid_len = len(spec)
    invalid_vec = np.zeros(50000, dtype=np.float32)
    invalid_vec[0] = 1
    vec = np.zeros((100, 50000), dtype=np.float32)
    for idx in range(100):
        if idx < valid_len:
            vec[idx, :] = mz_to_vec(spec[idx])
        else:
            vec[idx, :] = invalid_vec

    return vec


def get_data_dict(data_df):
    data_list = np.array(data_df).tolist()
    data_dict = {}
    gts = {}
    tokens = []
    token_dict = {}
    mz = {}
    for idx, data in tqdm(enumerate(data_list)):
        smi = data[0]
        ms2_list = data[4]

        token = [i for i in smi]
        gts[str(idx)] = smi
        tokens.extend(token)
        token_dict[str(idx)] = token
        ms2_list = [float(i) for i in ms2_list.split(' ')]
        mz[str(idx)] = ms2_list

    total_idx = range(len(data_list))
    train_idx = random.sample(total_idx, len(data_list) - 2000)
    val_test_idx = list(set(total_idx) - set(train_idx))
    val_idx = random.sample(val_test_idx, 1000)
    test_idx = list(set(val_test_idx) - set(val_idx))

    data_dict['gts'] = gts
    data_dict['token'] = token_dict
    data_dict['mz'] = mz

    sample_dict = {}
    sample_dict['train'] = train_idx
    sample_dict['val'] = val_idx
    sample_dict['test'] = test_idx

    data_dict['data'] = sample_dict

    idx_to_token = {}
    token_to_idx = {}
    tokens = list(set(tokens))
    base_tokens = ['<pad>', '<eos>', '<sos>']
    base_tokens.extend(tokens)

    for idx, token in enumerate(base_tokens):
        idx_to_token[str(idx)] = token
        token_to_idx[token] = str(idx)

    data_dict['idx_to_token'] = idx_to_token
    data_dict['token_to_idx'] = token_to_idx

    return data_dict


def get_data_dict_from_exp(data_list):
    data_dict = {}
    gts = {}
    tokens = []
    token_dict = {}
    mz = {}
    for idx, data in tqdm(enumerate(data_list)):
        smi = data[0]
        ms2_list = data[1]

        token = [i for i in smi]
        gts[str(idx)] = smi
        tokens.extend(token)
        token_dict[str(idx)] = token
        ms2_list = [float(i) for i in ms2_list.split(' ')]
        mz[str(idx)] = ms2_list

    data_dict['gts'] = gts
    data_dict['token'] = token_dict
    data_dict['mz'] = mz

    return data_dict


if __name__ == '__main__':
    file_path = "模型复现/lipid/MassGenie/medium_filtered.csv"
    data_dict = get_data_dict(load_data(file_path))

    with open("模型复现/lipid/MassGenie/data_dict.pkl", 'wb') as f:
        pickle.dump(data_dict, f)

