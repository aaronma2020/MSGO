import os
import subprocess
import pickle
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
from train import load_data
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


def get_predictions(file_path):
    with open(file_path, 'rb') as f:
        candidate_dict = pickle.load(f)

    return candidate_dict


def get_test_spec(file_path):
    data_df = pd.read_csv(file_path)
    mz = data_df['mz'].tolist()
    intensity = data_df['intensity'].tolist()

    mz_dict = {}
    for idx, data in enumerate(mz):
        mz_dict[str(idx)] = [float(i) for i in data.split(' ')]

    intensity_dict = {}
    for idx, data in enumerate(intensity):
        intensity_dict[str(idx)] = [float(i) for i in data.split(' ')]

    return mz_dict, intensity_dict


def check_smiles(smi):

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    else:
        return True


def filter_candidates_by_smi(candidates):
    filtered_candidates = []
    for smi in candidates:
        if check_smiles(smi) is True:
            filtered_candidates.append(smi)

    return filtered_candidates


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


def write_parameter_file():
    lines = [
        'PeakListPath = ./metfrag/input/peaklist.txt\n',
        'LocalDatabasePath = ./metfrag/input/candidates.csv\n',
        'MetFragDatabaseType = LocalCSV\n',
        'NeutralPrecursorMass = 500.0\n',
        'FragmentPeakMatchAbsoluteMassDeviation = 500.0\n',
        'FragmentPeakMatchRelativeMassDeviation = 1000000\n',
        'PrecursorIonMode = 1\n',
        'IsPositiveIonMode = True\n',
        'MetFragScoreTypes = FragmenterScore\n',
        'MetFragScoreWeights = 1.0\n',
        'MetFragCandidateWriter = CSV\n',
        'SampleName = result\n',
        'ResultsPath = ./metfrag/output\n',
        'MaximumTreeDepth = 2\n',
    ]
    with open(r"metfrag/input/parameter.txt", 'w') as f:
        f.writelines(lines)


def write_peaklist_file(mz, intensity):
    lines = []
    for peak, abundance in zip(mz, intensity):
        line = '\t'.join([str(peak), str(abundance)])
        line += '\n'
        lines.append(line)

    with open(r"metfrag/input/peaklist.txt", 'w') as f:
        f.writelines(lines)


def write_candidates_file(candidates):
    database_list = []
    columns = ['Identifier', 'InChI', 'MolecularFormula', 'SMILES']
    for idx, smi in enumerate(candidates):
        mol = Chem.MolFromSmiles(smi)
        inchi = Chem.MolToInchi(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        valid_smi = Chem.MolToSmiles(mol)
        database_list.append([str(idx), inchi, formula, valid_smi])

    df = pd.DataFrame(database_list, columns=columns)
    df.to_csv(r"metfrag/input/candidates.csv", index=None)


def get_metfrag_result():
    data_df = pd.read_csv(r"metfrag/output/result.csv")
    metfrag_result = data_df['SMILES'].tolist()

    return metfrag_result


def score_candidates_by_metfrag(data_dict):
    os.putenv("JAVA_HOME", r"D:\openjdk\jdk-23")
    os.environ["PATH"] += os.pathsep + r"D:\openjdk\jdk-23\bin"
    new_data_dict = {}
    for i in tqdm(range(data_dict.__len__())):
        idx = str(i)
        ground_truth = data_dict[idx]['ground_truth']
        candidates = data_dict[idx]['filtered_candidates']
        mz = data_dict[idx]['mz']
        intenstiy = data_dict[idx]['intensity']

        write_peaklist_file(mz, intenstiy)
        write_candidates_file(candidates)
        os.system('java -jar MetFrag2.6.1-CL.jar parameter.txt')

        metfrag_candidates = get_metfrag_result()

        new_data_dict[idx] = {'ground_truth': ground_truth,
                              'candidates': candidates,
                              'metfrag_candidates': metfrag_candidates,
                              }

        os.remove("metfrag/input/candidates.csv")
        os.remove("metfrag/input/peaklist.txt")

    return new_data_dict


def main():
    test_result_dict = get_predictions("lipid_test_result/1/test_result_dict.pkl")
    exp_result_dict = get_predictions("lipid_test_result/1/exp_result_dict.pkl")
    test_data_dict = load_data("模型复现/lipid/MassGenie/data_dict.pkl")
    exp_data_dict = load_data("模型复现/lipid/MassGenie/exp_dict.pkl")

    mz_dict, intensity_dict = get_test_spec("模型复现/lipid/MassGenie/medium_filtered.csv")

    test_ids = test_data_dict['data']['test']

    exp_dict = {}
    for i in range(exp_result_dict.__len__()):
        idx = str(i)
        ground_truth = exp_result_dict[idx]['ground_truth']
        candidates = exp_result_dict[idx]['candidates']
        candidates = filter_candidates_by_smi(candidates)
        mz = exp_data_dict['mz'][idx]
        intensity = [10 for _ in range(len(mz))]
        exp_dict[idx] = {'ground_truth': ground_truth,
                         'filtered_candidates': candidates,
                         'mz': mz,
                         'intensity': intensity}

    test_dict = {}
    for i in range(test_result_dict.__len__()):
        idx = str(i)
        ground_truth = test_result_dict[idx]['ground_truth']
        candidates = test_result_dict[idx]['candidates']
        candidates = filter_candidates_by_smi(candidates)
        mz = mz_dict[str(test_ids[i])]
        intensity = intensity_dict[str(test_ids[i])]
        test_dict[idx] = {'ground_truth': ground_truth,
                          'filtered_candidates': candidates,
                          'mz': mz,
                          'intensity': intensity}

    exp_dict_with_metfrag = score_candidates_by_metfrag(exp_dict)
    test_dict_with_metfrag = score_candidates_by_metfrag(test_dict)

    return exp_dict_with_metfrag, test_dict_with_metfrag


if __name__ == '__main__':
    exp_dict, test_dict = main()
    with open('lipid_test_result/1/lipid_exp_result_dict_with_metfrag.pkl', 'wb') as f:
        pickle.dump(exp_dict, f)
    with open('lipid_test_result/1/lipid_test_result_dict_with_metfrag.pkl', 'wb') as f:
        pickle.dump(test_dict, f)

