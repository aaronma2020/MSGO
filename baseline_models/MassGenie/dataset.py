import torch
import pickle
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    def __init__(self, src, tgt, tgt_y):
        super(TransformerDataset, self).__init__()
        self.src = src
        self.tgt = tgt
        self.tgt_y = tgt_y

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], self.tgt_y[index]

    def __len__(self):
        return self.src.size(0)


def get_data(data_dict, dataset='train', max_len=100):
    vocab = data_dict['token_to_idx']
    tokens = data_dict['token']
    mz = data_dict['mz']
    data_idx = data_dict['data'][dataset]

    src_list = []
    tgt_list = []
    tgt_y_list = []
    for idx in data_idx:
        token = tokens[str(idx)]
        token_list = ['<sos>']
        token_list.extend(token)
        token_list.append('<eos>')

        while len(token_list) < max_len:
            token_list.append('<pad>')

        ms_list = mz[str(idx)]
        src = []
        for i in range(max_len):
            if i < len(ms_list):
                ms = int(round(ms_list[i], 2) * 100)
                if ms > 50000:
                    src.append(0)
                else:
                    src.append(ms)
            else:
                src.append(0)

        token_list = [int(vocab[i]) for i in token_list]
        tgt = token_list[:-1]
        tgt_y = token_list[1:]

        src_list.append(src)
        tgt_list.append(tgt)
        tgt_y_list.append(tgt_y)

    src = torch.LongTensor(src_list)
    tgt = torch.LongTensor(tgt_list)
    tgt_y = torch.LongTensor(tgt_y_list)

    return src, tgt, tgt_y


def get_data_from_exp(data_dict, max_len=100):
    with open("模型复现/lipid/MassGenie/data_dict.pkl", 'rb') as f:
        all_data_dict = pickle.load(f)
    vocab = all_data_dict['token_to_idx']
    tokens = data_dict['token']
    mz = data_dict['mz']

    src_list = []
    tgt_list = []
    tgt_y_list = []
    for idx in range(data_dict['gts'].__len__()):
        token = tokens[str(idx)]
        token_list = ['<sos>']
        token_list.extend(token)
        token_list.append('<eos>')
        token_list = token_list[:100]

        while len(token_list) < max_len:
            token_list.append('<pad>')

        ms_list = mz[str(idx)]
        src = []
        for i in range(max_len):
            if i < len(ms_list):
                ms = int(round(ms_list[i], 2) * 100)
                if ms > 50000:
                    src.append(0)
                else:
                    src.append(ms)
            else:
                src.append(0)

        token_list = [int(vocab[i]) for i in token_list]
        tgt = token_list[:-1]
        tgt_y = token_list[1:]

        src_list.append(src)
        tgt_list.append(tgt)
        tgt_y_list.append(tgt_y)

    src = torch.LongTensor(src_list)
    tgt = torch.LongTensor(tgt_list)
    tgt_y = torch.LongTensor(tgt_y_list)

    return src, tgt, tgt_y


if __name__ == '__main__':
    # with open("模型复现\PFAS_data\MassGenie\data_dict.pkl", 'rb') as f:
    #     data_dict = pickle.load(f)

    # train_src, train_tgt, train_tgt_y = get_data(data_dict, 'train')
    # train_set = TransformerDataset(train_src, train_tgt, train_tgt_y)

    # from model import TransformerModel

    # model = TransformerModel(1024, 50000, 22)
    # src = train_set[:1][0]
    # tgt = train_set[:1][1]
    # out = model(src, tgt)

    pass



