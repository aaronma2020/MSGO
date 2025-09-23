import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import random
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.ML.Descriptors import MoleculeDescriptors
from decoder.moses.trans.model import TranslationModel
from decoder.moses.trans.trainer import TranslationTrainer
from decoder.moses.trans.config import get_parser
from torch.utils.data import Dataset, DataLoader


def get_descriptors_from_smiles(smiles):
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    mol = Chem.MolFromSmiles(smiles)
    des_list = ['MolLogP', 'MolMR', 'NumValenceElectrons', 'NumHAcceptors', 'NumHDonors', 'BalabanJ', 'TPSA', 'qed']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    result = [float(i) for i in list(calculator.CalcDescriptors(mol))]
    sa = sascorer.calculateScore(mol)
    result.append(sa)

    return result


def collate_func(batch_dic):
    batch_len = len(batch_dic)
    max_seq_length=max([dic['length'] for dic in batch_dic])
    mask_batch = torch.zeros((batch_len, max_seq_length))
    all_batch = []
    for i in range(len(batch_dic)):
        dic = batch_dic[i]
        mask_batch[i, :dic['length']] = 1
        all_batch.append([dic['x'], dic['label1'], dic['label2'], dic['length'], mask_batch])
    all_batch = sorted(all_batch, key=lambda x: x[3], reverse=True)
    x_batch = [i[0] for i in all_batch]
    label1_batch = [i[1]for i in all_batch]
    label2_batch = [i[2]for i in all_batch]
    mask_batch = [i[3] for i in all_batch]
    res = {}
    res['x'] = x_batch
    res['label1'] = label1_batch
    res['label2'] = label2_batch
    res['mask'] = mask_batch

    return res


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = [i[1] for i in data]
        self.label1 = [i[0] for i in data]
        self.label2 = [i[2] for i in data]

    def __getitem__(self, i):
        return {'x':self.data[i], 'label1':self.label1[i], 'label2':self.label2[i], 'length':len(self.data[i])}

    def __len__(self):
        return len(self.data)


class Regression_Model(nn.Module):
    def __init__(self):
        super(Regression_Model, self).__init__()
        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)

        return output


if __name__ == '__main__':
    data = joblib.load(r"E:\BQY\模型复现\lipid\Spec2Mol\smiles_encoder_train_data.joblib")
    data_list = np.array(data).tolist()
    smiles_list = np.array(data['smiles']).tolist()
    parser = get_parser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--save_frequency', type=int, default=1)
    parser.add_argument('--best_pct_match', type=float, default=0.0)
    parser.add_argument('--model_save', type=str, default=r'E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encoder_model')
    config = parser.parse_args()
    trainer = TranslationTrainer(config)
    vocab = trainer.get_vocabulary(smiles_list)
    model = TranslationModel(vocab=vocab, config=config)
    model.to('cuda:2')
    device_ids = [0, 1]
    # train_model = nn.DataParallel(model, device_ids=device_ids)
    regression_model = Regression_Model()
    train_list = [i for i in data_list]
    train_data = [[model.string2tensor(i[0]), model.string2tensor(i[1]), torch.tensor(i[2])] for i in train_list]
    train_dataset = MyDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_func)
    regression_model.to('cuda:1')
    trainer._train(model=model, regression_models=regression_model, train_loader=train_loader)
    # train_loader包含三列数据，第一列为random-smiles，第二列为canonical-smiles，第三列为9种物理化学性质的列表






