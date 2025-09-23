import joblib
import sys
import os
import pandas as pd
from SmilesEnumerator import SmilesEnumerator
from rdkit import RDConfig
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import QED
from rdkit import Chem
from sascorer import calculateScore
from tqdm import tqdm


def get_raw_data(file_path):
    return pd.read_csv((file_path))['smiles']


def get_randomize_smiles(smi):
    sme = SmilesEnumerator()
    return sme.randomize_smiles(smi)


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


def get_properties(smi):
    mol = Chem.MolFromSmiles(smi)
    logP = Descriptors.MolLogP(mol)
    molar_refractivity = Descriptors.MolMR(mol)
    number_of_valence_electrons = Descriptors.NumValenceElectrons(mol)
    number_of_hydrogen_bond_donors = Lipinski.NumHDonors(mol)
    number_of_hydrogen_bond_acceptors = Lipinski.NumHAcceptors(mol)
    balaban_J_value = GraphDescriptors.BalabanJ(mol)
    topological_polar_surface_area = rdMolDescriptors.CalcTPSA(mol)
    qed_drug_likeliness = QED.qed(mol)
    synthetic_accessibility = calculateScore(mol)

    return [logP,
            molar_refractivity,
            number_of_valence_electrons,
            number_of_hydrogen_bond_donors,
            number_of_hydrogen_bond_acceptors,
            balaban_J_value,
            topological_polar_surface_area,
            qed_drug_likeliness,
            synthetic_accessibility
            ]


def save_data(df, file_path):
    joblib.dump(df, file_path)


def main():
    raw_data_path = r'E:\BQY\模型复现\lipid\lipid\high.csv'
    save_data_path = r'E:\BQY\模型复现\lipid\Spec2Mol\smiles_encoder_train_data.joblib'
    smiles_list = get_raw_data(raw_data_path).to_list()
    randomize_smi_list = [get_randomize_smiles(i) for i in smiles_list]
    properties_list = [get_descriptors_from_smiles(i) for i in tqdm(smiles_list)]
    result_data = [[smiles_list[i], randomize_smi_list[i], properties_list[i]] for i in range(len(smiles_list))]
    result_df = pd.DataFrame(result_data, columns=['smiles', 'randomize_smiles', 'properties'])
    save_data(result_df, save_data_path)


if __name__ == '__main__':
    main()
