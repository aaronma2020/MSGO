import pandas as pd
import torch

from smiles_encoder_retrain import *
from utils import normalize_smiles

if __name__ == '__main__':
    high_dict = dict(torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encode_data\lipid_smiles_high_energy_dict.pt"))
    smiles_list = [i[0] for i in high_dict.items()]
    parser = get_parser()
    config = parser.parse_args()
    trainer = TranslationTrainer(config)
    vocab = trainer.get_vocabulary(smiles_list)
    model = TranslationModel(vocab=vocab, config=config)
    state_dict = torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encoder_model\model9974.pt")
    model.load_state_dict(state_dict)
    model.to('cuda:2')
    model.eval()
    smiles_list = dict(torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encode_data\lipid_experiment_smiles_high_energy_dict.pt"))
    all_list = sorted(zip([model.string2tensor(i) for i in smiles_list], smiles_list), key=lambda x: x[0].shape[0], reverse=True)
    smiles_tensor_list = [i[0] for i in all_list]
    smiles_sorted_list = [i[1] for i in all_list]
    smiles_id_dict = {}
    for id, smiles in enumerate(smiles_sorted_list):
        smiles_id_dict[str(id)] = smiles
    torch.save(smiles_id_dict, r"E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encode_data\lipid_experiment_smiles_id_dict.pt")
    smiles_encode_dict = {}
    split = 100
    for i in range(len(smiles_tensor_list)//split + 1):
        smiles_encode = model.forward_encoder(smiles_tensor_list[i*split:(i+1)*split])[2]
        smiles_encode = smiles_encode.detach().to('cpu')
        print(smiles_encode.device)
        for j in range(smiles_encode.shape[0]):
            id = i*split + j
            smiles = smiles_id_dict[str(id)]
            smiles_encode_dict[smiles] = smiles_encode[j]
    torch.save(smiles_encode_dict, r"E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encode_data\lipid_experiment_smiles_encode_dict.pt")








