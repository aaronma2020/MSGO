'''
Author: Zheng Ma
Date: 2022-02-21 11:21:53
LastEditTime: 2022-05-04 11:05:23
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/utils/eval.py

'''
import torch
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolHash
# from .progressbar import get_progressbar
from .loader import build_loader
from collections import Counter
import re
from tqdm import tqdm


def eval_real_beam1(model, opt, mz, mz_mask):
     with torch.no_grad():
        mz, mz_mask = mz.cuda(), mz_mask.cuda()
        seq = model.sample(mz, mz_mask, beam_size=1)
        sents = decode_sequence(opt.ix_to_token, seq.data)
        return sents[0]


def eval_real(model, opt, mz, mz_mask, formula_id=None, mol_mass=None, beam_size=10, smiles=None, formula=None, precusor=None, rule=None, rank=10):
    with torch.no_grad():
        mz, mz_mask = mz.cuda(), mz_mask.cuda()
        if formula_id != None:
            formula_id = formula_id.unsqueeze(-1).long().cuda()
        if mol_mass != None:
            mol_mass = mol_mass.unsqueeze(-1).long().cuda()
        
        seq = model.sample(mz, mz_mask, formula_id, mol_mass, beam_size)[0]

        if rule == 'formula':
            other_sents = []
            candidates = []
            for bm in range(beam_size):
                # sent = decode_sequence(opt.ix_to_token, seq.unsqueeze(0))[0]
                sent = decode_sequence(opt.ix_to_token, seq[bm]['seq'].unsqueeze(0))[0]
                try: 
                    cand_formula = Smilestoformula(sent)
                except:
                    continue

                if extra_atoms(cand_formula) == extra_atoms(formula):
                    candidates.append(sent)
                else:
                    other_sents.append(sent)

                if len(candidates) == rank:
                    break
            match = [0] * rank
            if len(candidates) != 0:
                match[:len(candidates)] = [1] * len(candidates)

            if len(candidates) == rank:
                pass
            else:
                candidates.extend(other_sents[:10-len(candidates)])

            ecfps, fcfps = [], []
            for c in candidates:
                e, f = ecfp_fcfp(smiles, c)
                ecfps.append(e)
                fcfps.append(f)

            return candidates, ecfps, fcfps, match
        elif rule == 'precusor':
            candidates = []
            for bm in range(beam_size):
                sent = decode_sequence(opt.ix_to_token, seq[bm]['seq'].unsqueeze(0))[0]
                try: 
                    cand_formula = Smilestoformula(sent)
                except:
                    continue
                cand_precursor = computer_MW_formula(extra_atoms(replace_halogen(Smilestoformula(sent))), atom2mass) - atom2mass['H'] + atom2mass['E']
                candidates.append([sent, abs(precusor-cand_precursor)])

            # sort
            candidates = sorted(candidates, key=lambda x: x[1])[:rank]          
            smiles_list = []
            mass_diff = []
            ecfps, fcfps = [], []

            if smiles != None:
                for c in candidates:
                    e, f = ecfp_fcfp(smiles, c[0])
                    ecfps.append(e)
                    fcfps.append(f)
                    smiles_list.append(c[0])
                    mass_diff.append(c[1])

                return smiles_list, ecfps, fcfps, mass_diff
            else:
                for c in candidates:
                    smiles_list.append(c[0])
                    mass_diff.append(c[1])
                
                return smiles_list, mass_diff 







        


            




            






def eval(model, dataloader, opt):
    beam_size = opt.beam_size
    predictions = []
    total_results = None
    # bar = get_progressbar(len(dataloader))
    
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            tmp = [data['mz'], data['mz_mask']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            mz, mz_mask = tmp
            if opt.use_formula:
                formula = data['formula'].cuda()
            else:
                formula = None
            if opt.use_mol_mass:
                mol_mass = data['mol_mass'].cuda()
            else:
                mol_mass = None
            seq = model.sample(mz, mz_mask, formula, mol_mass, beam_size)
            if beam_size == 1:
                sents = decode_sequence(opt.ix_to_token, seq.data)
            else:
                sents = [decode_sequence(opt.ix_to_token, _[0]['seq'].unsqueeze(0))[0] for _ in seq]

            for k, sent in enumerate(sents):
                entry = {'id': data['info'][k]['ix']}
                print('-'*20)
                print('real:', data['info'][k]['gts'])
                print('predict:', sent)
                metric_results = metric(sent, data['info'][k]['gts'])
                entry.update(metric_results)
                predictions.append(entry)
                if not total_results:
                    total_results = metric_results
                else:
                    for m, s in metric_results.items():
                        total_results[m] += s
                
            # bar.update(i)
    for k, v in total_results.items():
        total_results[k] = v / len(predictions)
    return total_results

def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt.replace('@@ ', ''))
    return out


def metric(cand, gt, metric_kwargs={'ecfp_fcfp': 1, 'accurate':1}):

    return_dict = {}
    if metric_kwargs['ecfp_fcfp']:
        ecfp, fcfp = ecfp_fcfp(cand, gt)
        return_dict['ecfp'] = ecfp
        return_dict['fcfp'] = fcfp
    if metric_kwargs['accurate']:
        acc = int(cand == gt)
        return_dict['accurate'] = acc

    return return_dict


def ecfp_fcfp(smile1, smile2):
    try:
        smi_1 = Chem.MolFromSmiles(smile1)
        smi_2 = Chem.MolFromSmiles(smile2)
        ecfp_1 = AllChem.GetMorganFingerprint(smi_1, 4)
        ecfp_2 = AllChem.GetMorganFingerprint(smi_2, 4)
        fcfp_1 = AllChem.GetMorganFingerprint(smi_1, 4, useFeatures=True)
        fcfp_2 = AllChem.GetMorganFingerprint(smi_2, 4, useFeatures=True)
        ecfp_similarity = DataStructs.DiceSimilarity(ecfp_1, ecfp_2)
        fcfp_similarity = DataStructs.DiceSimilarity(fcfp_1, fcfp_2)
    except:
        return 0, 0
    return ecfp_similarity, fcfp_similarity
    
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
    '''计算部分分子式  对应的质量数'''
    atoms = frament_dict.keys()
    mass = 0
    for atom in atoms:
        mass += atom2mass[atom] * int(frament_dict[atom])
    return mass


def Smilestoformula(smi):
    '''通过smiles计算分子式'''
    mol = Chem.MolFromSmiles(smi)
    return rdMolHash.MolHash(mol, rdMolHash.HashFunction.MolFormula)

def extra_atoms(formula, smiles=None):
    '''提取分子式各个原子的个数'''
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
    '''将cl br替换，便于切片处理'''
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string