import pickle

from utils import get_data_dict_from_exp
from dataset import get_data_from_exp


def read_exp_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data_list = []
    temp = []
    for line in lines:
        if line == '\n':
            data_list.append(temp)
            temp = []
        else:
            temp.append(line.replace('\n', ''))

    spec_list = []
    for data in data_list:
        spec = ' '.join([i.split(' ')[0] for i in data[2:]])
        spec_list.append(spec)

    smiles_list = [i[0].split(' ')[1] for i in data_list]
    mz_list = [float(i[1]) for i in data_list]
    data_list = zip(smiles_list, spec_list, mz_list)
    data_list = [[i[0], i[1]] for i in data_list if i[2] <= 500]
    data_list = data_list[1:]
    error_ids = [i[0] for i in enumerate(mz_list) if i[1] > 500]
    error_ids.append(0)
    error_ids = sorted(error_ids)
    data_list.append(data_list[0])

    return data_list, error_ids


if __name__ == '__main__':
    file_path = "模型复现/PFAS_data/PFAS experiment mass spectra.txt"
    data_list, error_ids = read_exp_data(file_path)
    data_dict = get_data_dict_from_exp(data_list)
    src, tgt, tgt_y = get_data_from_exp(data_dict)
    with open("模型复现/PFAS_data/MassGenie/exp_dict.pkl", 'wb') as f:
        pickle.dump(data_dict, f)

