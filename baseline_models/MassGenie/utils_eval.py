from rdkit import Chem
from rdkit import RDLogger
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
import torch
import pickle

RDLogger.DisableLog('rdApp.*')


def get_monoisotopic_mass(smi):
    mol = Chem.MolFromSmiles(smi)
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    return mw


def check_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    else:
        return True


def get_formula(smi):
    return rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smi))


def filter_candidates_by_formula(true_smi, candidates):
    filtered_candidates = []
    true_mol = Chem.MolFromSmiles(true_smi)
    true_formula = rdMolDescriptors.CalcMolFormula(true_mol)
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        if formula == true_formula:
            filtered_candidates.append(smi)

    return filtered_candidates


def get_ecfp(smi1, smi2):
    fpgen = AllChem.GetMorganGenerator(radius=2)
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    fp1 = fpgen.GetSparseCountFingerprint(m1)
    fp2 = fpgen.GetSparseCountFingerprint(m2)

    return DataStructs.DiceSimilarity(fp1,fp2)


def get_fcfp(smi1, smi2):
    invgen = AllChem.GetMorganFeatureAtomInvGen()
    fpgen = AllChem.GetMorganGenerator(radius=2, atomInvariantsGenerator=invgen)
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    fp1 = fpgen.GetSparseCountFingerprint(m1)
    fp2 = fpgen.GetSparseCountFingerprint(m2)

    return DataStructs.DiceSimilarity(fp1,fp2)


def evaluate_spec2mol_result(data_dict, data_type='test'):
    formula_count = 0
    smiles_count = 0
    avg_ecfp_score = 0.0
    avg_fcfp_score = 0.0
    top10_list = []
    for ground_truth, candidates in data_dict.items():
        if data_type == 'exp':
            ground_truth = ground_truth.split(' ')[1]
        ground_truth = Chem.MolToSmiles(Chem.MolFromSmiles(ground_truth))
        candidates = list(set(candidates))
        candidates = [i for i in candidates if check_smiles(i) is True]
        candidates = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in candidates]
        ground_truth_mw = get_monoisotopic_mass(ground_truth)
        mw_list = [abs(get_monoisotopic_mass(i) - ground_truth_mw) for i in candidates]
        candidates = [i[1] for i in sorted(zip(mw_list, candidates), key=lambda x: x[0])]

        top10_list.append(candidates[:10])

        pred = candidates[0]
        ground_truth_formula = get_formula(ground_truth)
        if get_formula(pred) == ground_truth_formula:
            formula_count += 1
            if pred == ground_truth:
                smiles_count += 1
        ecfp_score = get_ecfp(ground_truth, pred)
        fcfp_score = get_fcfp(ground_truth, pred)

        avg_ecfp_score += ecfp_score
        avg_fcfp_score += fcfp_score

    eval_dict = {
        'top10': top10_list,
        'true_formula_rate': formula_count / data_dict.__len__(),
        'true_smiles_rate': smiles_count / data_dict.__len__(),
        'ecfp': avg_ecfp_score / data_dict.__len__(),
        'fcfp': avg_fcfp_score / data_dict.__len__()
    }

    return eval_dict


def evaluate_massgenie_result(data_dict):
    formula_count = 0
    smiles_count = 0
    avg_ecfp_score = 0.0
    avg_fcfp_score = 0.0
    top10_list = []
    for i in range(data_dict.__len__()):
        idx = str(i)
        ground_truth = data_dict[idx]['ground_truth']
        candidates = data_dict[idx]['metfrag_candidates']
        ground_truth = Chem.MolToSmiles(Chem.MolFromSmiles(ground_truth))
        candidates = [i for i in candidates if check_smiles(i) is True]
        candidates = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in candidates]
        filtered_candidates = filter_candidates_by_formula(ground_truth, candidates)
        if len(filtered_candidates) == 0:
            pred = candidates[0]
            top10_list.append(candidates[:10])
        else:
            pred = filtered_candidates[0]
            if len(filtered_candidates) >= 10:
                top10_list.append(filtered_candidates[:10])
            else:
                filtered_candidates.extend(candidates)
                top10_list.append(filtered_candidates[:10])

        ground_truth_formula = get_formula(ground_truth)
        if get_formula(pred) == ground_truth_formula:
            formula_count += 1
            if pred == ground_truth:
                smiles_count += 1
        ecfp_score = get_ecfp(ground_truth, pred)
        fcfp_score = get_fcfp(ground_truth, pred)

        avg_ecfp_score += ecfp_score
        avg_fcfp_score += fcfp_score

    eval_dict = {
        'top10': top10_list,
        'true_formula_rate': formula_count / data_dict.__len__(),
        'true_smiles_rate': smiles_count / data_dict.__len__(),
        'ecfp': avg_ecfp_score / data_dict.__len__(),
        'fcfp': avg_fcfp_score / data_dict.__len__()
    }

    return eval_dict


if __name__ == '__main__':
    spec2mol_test_dict = dict(torch.load("Spec2Mol/test_result/direct_test_candidate_dict.pt"))
    spec2mol_exp_dict = dict(torch.load("Spec2Mol/test_result/direct_exp_candidate_dict.pt"))
    spec2mol_test_eval = evaluate_spec2mol_result(spec2mol_test_dict, 'test')
    spec2mol_exp_eval = evaluate_spec2mol_result(spec2mol_exp_dict, 'exp')

    with open("test_result/test_result_dict_with_metfrag.pkl", 'rb') as f:
        massgenie_test_dict = pickle.load(f)
    with open("test_result/exp_result_dict_with_metfrag.pkl", 'rb') as f:
        massgenie_exp_dict = pickle.load(f)
    massgenie_test_eval = evaluate_massgenie_result(massgenie_test_dict)
    massgenie_exp_eval = evaluate_massgenie_result(massgenie_exp_dict)

