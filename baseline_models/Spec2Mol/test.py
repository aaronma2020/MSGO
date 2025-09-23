import argparse
import torch
import numpy as np
from model1D2Conv import Net1D
from smiles_encoder_retrain import *
from utils_eval import *
from dataset import MSDataset_4channels
from tqdm import tqdm
from rdkit import Chem, RDLogger


def read_experiment_spec(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    temp = []
    spec_list = []
    for line in lines:
        if line == '\n':
            spec_list.append(temp)
            temp = []
        else:
            temp.append(line.replace('\n', ''))


def generate_candidate_list(candidate_length=128):
    encoder_parser = argparse.ArgumentParser()
    encoder_parser.add_argument('-num_epochs', type=int, default=201, help='Number of epochs')
    encoder_parser.add_argument('-lr', type=float, default=0.004, help='Learning rate')
    encoder_parser.add_argument('-reg', type=float, default=0.0001, help='Weight decay')
    encoder_parser.add_argument('-batch_size', type=int, default=512, help='Batch size')
    encoder_parser.add_argument('-valid_size', type=int, default=512, help='Number of molecules in validset')
    encoder_parser.add_argument('-batch_size_valid', type=int, default=128, help='Batch size for valid')
    encoder_parser.add_argument('-channels_med_1', type=int, default=4, help='Number of channels after first conv layer')
    encoder_parser.add_argument('-channels_out', type=int, default=8, help='Number of output channels in the last conv layer')
    encoder_parser.add_argument('-conv_kernel_dim_1', type=int, default=200, help='Kernel size for first conv layer')
    encoder_parser.add_argument('-conv_kernel_dim_2', type=int, default=200, help='Kernel size for second conv layer')
    encoder_parser.add_argument('-conv_stride_1', type=int, default=1, help='Stride for first conv layer')
    encoder_parser.add_argument('-conv_stride_2', type=int, default=1, help='Stride for second conv layer')
    encoder_parser.add_argument('-conv_dilation', type=int, default=1, help='Dilation for conv layers')
    encoder_parser.add_argument('-conv_padding_1', type=int, default=100, help='Padding for first conv layer')
    encoder_parser.add_argument('-conv_padding_2', type=int, default=100, help='Padding for second conv layer')
    encoder_parser.add_argument('-pool_kernel_dim_1', type=int, default=50, help='Kernel size for first pool layer')
    encoder_parser.add_argument('-pool_kernel_dim_2', type=int, default=50, help='Kernel size for second pool layer')
    encoder_parser.add_argument('-pool_stride_1', type=int, help='Stride for first pool layer')
    encoder_parser.add_argument('-pool_stride_2', type=int, help='Stride for second pool layer')
    encoder_parser.add_argument('-pool_dilation', type=int, default=1, help='Dilation for pool layers')
    encoder_parser.add_argument('-pool_padding_1', type=int, default=0, help='Padding for first pool layers')
    encoder_parser.add_argument('-pool_padding_2', type=int, default=0, help='Padding for second pool layers')
    encoder_parser.add_argument('-fc_dim_1', type=int, default=512, help='Size of the fully connected layer')
    encoder_parser.add_argument('-conv_layers', type=int, default=2, help='Number of conv layers')
    encoder_parser.add_argument('-resolution', type=int, default=2, help='Number of decimal points for mass')
    encoder_parser.add_argument('-save_embs', type=bool, default=False, help='Save embeddings for train/valid/test data')
    encoder_parser.add_argument('-save_models', type=bool, default=True, help='Save trained models')
    encoder_parser.add_argument('-neg', type=bool, default=False, help='Use negative mode spectra if true')
    encoder_parser.add_argument('-emb_dim', type=int, default=512, help='Dimensionality of the embedding space')
    encoder_parser.add_argument('-augm', type=bool, default=False, help='Perorm data augmentation if true')
    encoder_args = encoder_parser.parse_args()
    encoder_args.pool_stride_1 = encoder_args.pool_kernel_dim_1
    encoder_args.pool_stride_2 = encoder_args.pool_kernel_dim_2
    encoder_args.channels_in = 2
    test_set = MSDataset_4channels('valid', encoder_args.resolution, encoder_args.neg)
    exp_set = MSDataset_4channels('experiment', encoder_args.resolution, encoder_args.neg)
    data_dim = test_set.get_data_dim()[1]

    encoder = Net1D(encoder_args, data_dim).cuda()
    encoder_state_dict = torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_spectra_encoder_model\105.pt")
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    high_dict = dict(torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encode_data\lipid_smiles_high_energy_dict.pt"))
    smiles_encode_dict = dict(torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encode_data\lipid_smiles_encode_dict.pt"))
    smiles_list = [i[0] for i in high_dict.items()]
    decoder_parser = get_parser()
    config = decoder_parser.parse_args()
    trainer = TranslationTrainer(config)
    vocab = trainer.get_vocabulary(smiles_list)
    decoder = TranslationModel(vocab=vocab, config=config).cuda()
    decoder_state_dict = torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_smiles_encoder_model\model9974.pt")
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()

    test_loader = DataLoader(test_set,
                             batch_size=64)
    exp_loader = DataLoader(exp_set,
                            batch_size=64)

    total_test_dict = {}
    total_exp_dict = {}
    for j in tqdm(range(candidate_length)):
        temp_smiles_order = []
        test_result = []

        for data in test_loader:
            spectra = data['spectra'].float().cuda()
            embedding = data['embedding'].cuda()
            smiles = data['smiles']
            z = encoder(spectra)
            result = decoder.sample(n_batch=len(z), z=z)
            temp_smiles_order.extend(smiles)
            test_result.extend(result)

        for i in range(len(temp_smiles_order)):
            smi = temp_smiles_order[i]
            if j == 0:
                total_test_dict[smi] = [test_result[i]]
            else:
                total_test_dict[smi].append(test_result[i])

        temp_smiles_order = []
        exp_result = []
        for data in exp_loader:
            spectra = data['spectra'].float().cuda()
            embedding = data['embedding'].cuda()
            smiles = data['smiles']
            z = encoder(spectra)
            result = decoder.sample(n_batch=len(z), z=z)
            temp_smiles_order.extend(smiles)
            exp_result.extend(result)

        for i in range(len(temp_smiles_order)):
            smi = temp_smiles_order[i]
            if j == 0:
                total_exp_dict[smi] = [exp_result[i]]
            else:
                total_exp_dict[smi].append(exp_result[i])

    return total_test_dict, total_exp_dict


def get_monoisotopic_mass(smi):
    mol = Chem.MolFromSmiles(smi)
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    return mw


def eval_candidate_dict(candidate_dict, method):
    RDLogger.DisableLog('rdApp.*')
    mw_all = []
    smi_length_all = []

    correct_count = 0
    formula_count = 0
    dmw_min_list = []
    dmw_avg_list = []
    dmf_min_list = []
    dmf_avg_list = []
    cosine_max_list = []
    cosine_avg_list = []

    max_mcs_ratio_list = []
    max_mcs_tan_list = []
    max_mcs_coef_list = []
    avg_mcs_ratio_list = []
    avg_mcs_tan_list = []
    avg_mcs_coef_list = []

    for true_smi, pred_list in tqdm(candidate_dict.items()):
        if method == 'exp':
            true_smi = true_smi.split(' ')[1]
        true_mw = get_monoisotopic_mass(true_smi)
        mw_all.append(true_mw)
        smi_length_all.append(len(true_smi))

        pred_list = list(get_valid(pred_list))
        mw_list = [get_monoisotopic_mass(i) for i in pred_list]
        mw_list = [abs(i - true_mw) for i in mw_list]
        pred_list = [i[1] for i in sorted(zip(mw_list, pred_list), key=lambda x: x[0])][:20]

        if true_smi in pred_list:
            correct_count += 1
        if compare_formulas(pred_list, true_smi) is True:
            formula_count += 1

        dmw_min_list.append(get_MW_dif_min(pred_list, true_smi))
        dmw_avg_list.append(get_MW_dif_avg(pred_list, true_smi))
        dmf_min_list.append(get_formulas_min_distance(pred_list, true_smi))
        dmf_avg_list.append(get_formulas_avg_distance(pred_list, true_smi))
        cosine_max_list.append(get_max_cosine(pred_list, true_smi, 2, 1024)[1])
        cosine_avg_list.append(get_avg_cosine(pred_list, true_smi, 2, 1024))

        max_mcs_result = get_max_mcs(pred_list, true_smi)
        max_mcs_ratio_list.append(max_mcs_result[1])
        max_mcs_tan_list.append(max_mcs_result[2])
        max_mcs_coef_list.append(max_mcs_result[3])

        avg_mcs_result = get_avg_mcs(pred_list, true_smi)
        avg_mcs_ratio_list.append(avg_mcs_result[0])
        avg_mcs_tan_list.append(avg_mcs_result[1])
        avg_mcs_coef_list.append(avg_mcs_result[2])

    avg_mw = np.mean(mw_all)
    avg_smi_length = np.mean(smi_length_all)

    correct = correct_count / candidate_dict.__len__() * 100
    correct_formula = formula_count / candidate_dict.__len__() * 100

    dmw_min = np.mean(dmw_min_list)
    dmw_avg = np.mean(dmw_avg_list)
    dmf_min = np.mean(dmf_min_list)
    dmf_avg = np.mean(dmf_avg_list)
    cosine_max = np.mean(cosine_max_list)
    cosine_avg = np.mean(cosine_avg_list)

    max_mcs_ratio = np.mean(max_mcs_ratio_list)
    avg_mcs_ratio = np.mean(avg_mcs_ratio_list)
    max_mcs_tan = np.mean(max_mcs_tan_list)
    avg_mcs_tan = np.mean(avg_mcs_tan_list)
    max_mcs_coef = np.mean(max_mcs_coef_list)
    avg_mcs_coef = np.mean(avg_mcs_coef_list)

    eval_dict = {}
    eval_dict['avg_mw'] = avg_mw
    eval_dict['avg_smi_length'] = avg_smi_length
    eval_dict['correct'] = correct
    eval_dict['correct_formula'] = correct_formula
    eval_dict['dmw_min'] = dmw_min
    eval_dict['dmw_avg'] = dmw_avg
    eval_dict['dmf_min'] = dmf_min
    eval_dict['dmf_avg'] = dmf_avg
    eval_dict['cosine_max'] = cosine_max
    eval_dict['cosine_avg'] = cosine_avg
    eval_dict['max_mcs_ratio'] = max_mcs_ratio
    eval_dict['avg_mcs_ratio'] = avg_mcs_ratio
    eval_dict['max_mcs_tan'] = max_mcs_tan
    eval_dict['avg_mcs_tan'] = avg_mcs_tan
    eval_dict['max_mcs_coef'] = max_mcs_coef
    eval_dict['avg_mcs_coef'] = avg_mcs_coef

    return eval_dict


def eval_canonical_result(candidate_dict):
    count = 0
    formula_count = 0
    all_smiles = []
    for true_smi, pred_list in tqdm(candidate_dict.items()):
        temp = []
        true_smi = normalize_smiles(true_smi, True, False)
        pred_list = get_valid(pred_list)
        pred_list = [normalize_smiles(i, True, False) for i in pred_list]
        true_mw = get_monoisotopic_mass(true_smi)
        mw_list = []
        for i in pred_list:
            try:
                mass = get_monoisotopic_mass(i)
                mw_list.append([i, mass])
            except Exception:
                pass
        mw_list = [[i[0], abs(i[1] - true_mw)] for i in mw_list]
        pred_list = [i[1] for i in sorted(mw_list, key=lambda x: x[0]) if i[0] < 10]
        if true_smi in pred_list:
            count += 1
        if compare_formulas(pred_list, true_smi) is True:
            formula_count += 1
        temp.append(true_smi)
        temp.extend(pred_list)
        all_smiles.append(temp)
    print('correct_smiles_rate', count / candidate_dict.__len__())
    print('correct_formula_rate', formula_count / candidate_dict.__len__())

    return all_smiles


if __name__ == '__main__':
    test_candidate, exp_candidate = generate_candidate_list(500)
    torch.save(test_candidate, r"E:\BQY\PycharmProjects\Spec2Mol\lipid_test_result\lipid_direct_test_candidate_dict.pt")
    torch.save(exp_candidate, r"E:\BQY\PycharmProjects\Spec2Mol\lipid_test_result\lipid_direct_exp_candidate_dict.pt")

    '''
    test_candidate_dict = dict(torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_test_result\lipid_direct_test_candidate_dict.pt"))
    exp_candidate_dict = dict(torch.load(r"E:\BQY\PycharmProjects\Spec2Mol\lipid_test_result\lipid_direct_exp_candidate_dict.pt"))
    test_result = eval_candidate_dict(test_candidate_dict, method='test')
    exp_result = eval_candidate_dict(exp_candidate_dict, method='test')
    torch.save(test_result, r"E:\BQY\PycharmProjects\Spec2Mol\lipid_test_result\lipid_direct_test_result.pt")
    torch.save(exp_result, r"E:\BQY\PycharmProjects\Spec2Mol\lipid_test_result\lipid_direct_exp_result.pt")
    '''