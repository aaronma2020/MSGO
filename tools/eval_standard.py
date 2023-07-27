'''
Author: Zheng Ma
Date: 2022-02-22 13:00:11
LastEditTime: 2022-04-02 21:59:28
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/tools/eval_standard.py

'''

import torch
import pandas as pd
import argparse
import os
import sys
sys.path.append('.')
import models.TransModel as models
from utils.eval import eval_real, eval_real_beam1
import json
from rdkit import Chem
from rdkit.Chem import rdMolHash
import re
import argparse

atom2mass = {
    'H': 1.00783,
    'B': 11.00931,
    'C': 12.0,
    'N': 14.00307,
    'O': 15.99491,
    'F': 18.99840,
    'Si': 27.97693,
    'P': 30.97376,
    'S': 31.97207,
    'L': 34.96885,
    'R': 78.91834,
    'I': 126.90448,
    'E': 0.000548579908999924
}
def computer_MW_formula(frament_dict, atom2mass):
    atoms = frament_dict.keys()
    mass = 0
    for atom in atoms:
        mass += atom2mass[atom] * int(frament_dict[atom])
    return mass

def extra_atoms(formula, smiles=None):
    atom_dict = {}
    pattern = re.compile(r'[A-Z][a-z]?[0-9]*')
    items = pattern.findall(formula)
    for item in items:
        pattern = re.compile(r'[A-Z][a-z]?')
        atom = pattern.findall(item)[0]
        pattern = re.compile(r'[0-9]+')
        num = pattern.findall(item)
        if len(num) != 0:
            num = num[0]
        else:
            num = 1
        atom_dict[atom] = num
    
    return atom_dict

def replace_halogen(string):
    '''将cl br替换, 便于切片处理'''
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string

def smiles2formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    return formula


def extra_atoms(formula):
    atom_dict = {}
    pattern = re.compile(r'[A-Z][a-z]?[0-9]*')
    items = pattern.findall(formula)
    for item in items:
        pattern = re.compile(r'[A-Z][a-z]?')
        atom = pattern.findall(item)[0]
        pattern = re.compile(r'[0-9]+')
        num = pattern.findall(item)
        if len(num) != 0:
            num = num[0]
        else:
            num = 1
        atom_dict[atom] = num
    return atom_dict

def process_mz(mz, mz_to_ix):
    id_mz = []
    for m in mz:
        if m in list(mz_to_ix.keys()): id_mz.append(mz_to_ix[m])
        else: continue 
    return list(map(int, id_mz))

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', required=True, type=str)
parser.add_argument('--append', type=str, default='best')
parser.add_argument('--save_result', type=str, default='./results')
parser.add_argument('--real_csv', type=str, default='./data/lipid/real.csv')
parser.add_argument('--beam_size', type=int, default=500)
parser.add_argument('--precise', type=int, default=0)
parser.add_argument('--select_rule', type=str, default='formula', help='formula, precursor')
parser.add_argument('--out_csv', default='./results.csv')
parser.add_argument('--polar', default="pos")
eval_args = parser.parse_args()

def process_mz(mz, mz_to_ix):
    id_mz = []
    for m in mz:
        if str(m) in list(mz_to_ix.keys()): 
            if mz_to_ix[str(m)] not in id_mz:
                id_mz.append(mz_to_ix[str(m)])
        else: continue 
    return list(map(int, id_mz))


ckpt_path = os.path.join(eval_args.log_path, 'model'+eval_args.append+'.tar')
checkpoint = torch.load(ckpt_path)
opt = checkpoint['opt']
mz_to_ix = opt.mz_to_ix
model = models.TransModel(opt).cuda() 
model.load_state_dict(checkpoint['model'])
# eval
print('Testing')
model.eval()

real_file = pd.read_csv(eval_args.real_csv)
total_num = 0
rank1_ecfp, rank1_fcfp = 0, 0
rank1, rank3, rank5, rank10 = 0, 0, 0, 0
max_nums = 0
result = pd.DataFrame(columns=[])

print(f"data number: {len(real_file.index)}")
for index in real_file.index:
    smiles = real_file.loc[index, 'smiles']
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mz = real_file.loc[index, 'mz'].split()
    try:
        mz = [round(float(m), eval_args.precise) for m in mz]
    except:
        print("bad case")
        continue
    # precusor = float(real_file.loc[index, 'precusor_mz'])
    # formula = real_file.loc[index, 'formula']
    formula = smiles2formula(smiles)
    precursor = computer_MW_formula(extra_atoms(replace_halogen(formula)), atom2mass) - atom2mass['H'] + atom2mass['E'] 
    if opt.use_precursor:
        precursor_round = round(float(precursor), eval_args.precise) 
        mz.insert(0, precursor_round)

    mz = torch.Tensor(process_mz(mz, mz_to_ix)).long().unsqueeze(0)
    mz_mask = [1 for i in range(len(mz[0]))] 
    mz_mask = torch.Tensor(mz_mask).long().unsqueeze(0)

    if opt.use_formula:
        formula_id = torch.Tensor([opt.formula_to_ix.get(formula, opt.formula_to_ix['UNK'])])
    else:
        formula_id = None
    
    if opt.use_mol_mass:
        if eval_args.polar == "pos":
            mol_mass = round(precursor - atom2mass['H'] + atom2mass['E'], eval_args.precise)
        else:
            mol_mass = round(precursor + atom2mass['H'] - atom2mass['E'], eval_args.precise)
        mol_mass = torch.Tensor([opt.mol_mass_to_ix.get(str(mol_mass), opt.mol_mass_to_ix['UNK'])])
    else:
        mol_mass = None

    try:
        level = real_file.loc[index, 'level']
    except:
        level = 0
    # c = eval_real_beam1(model, opt, mz, mz_mask)
    # print(c)
    # print(smiles)
    # print('*'*20)
    # if c == smiles:
    #     rank1 += 1
    # continue

    candidates, ecfps, fcfps, match_num = eval_real(model, opt, mz, mz_mask, formula_id, mol_mass, \
                                    beam_size=eval_args.beam_size, \
                                    smiles=smiles, formula=formula, precusor=precursor, \
                                    rule=eval_args.select_rule
                                    )
    candidates = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in candidates]


    if smiles != None:
        r1 = smiles in candidates[:1]
        r3 = smiles in candidates[:3]
        r5 = smiles in candidates[:5]
        r10 = smiles in candidates[:10]

        rank1 += r1
        rank3 += r3
        rank5 += r5
        rank10 += r10
        if len(candidates) != 0:
            rank1_ecfp += ecfps[0]
            rank1_fcfp += fcfps[0]
        else:
            candidates = ['None']
            ecfps = [0]
            fcfps = [0]


        for i, c in enumerate(candidates):
            tmp = {'mz': real_file.loc[index, 'mz'], 'level': level, \
                'generate': c, 'real': smiles, 'match': match_num[i], 'ecfp': ecfps[i], 'fcfp': fcfps[i], \
                    'rank1':r1, 'rank3': r3, 'rank5':r5, 'rank10': r10, 'rank': i, \
                        }
            result = result.append(tmp, ignore_index=True)

    else:
        for i, c in enumerate(candidates):
            tmp = {'mz': real_file.loc[index, 'mz'], \
                'generate': c, 'rank': i}
            result = result.append(tmp, ignore_index=True)
    print('real:', smiles)
    print('generate:', candidates[0])
    total_num += 1
    print(f"down {total_num}")
result.to_csv(eval_args.out_csv)
print('total_num:', total_num)
print('rank1 ecfp:', rank1_ecfp / total_num)
print('rank1 fcfp:', rank1_fcfp / total_num)
print('rank1 num / rate:', rank1, rank1/total_num)
print('rank3 num / rate:', rank3, rank3/total_num)
print('rank5 num / rate:', rank5, rank5/total_num)
print('rank10 num / rate:', rank10, rank10/total_num)


    


    

    























