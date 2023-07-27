'''
Author: Zheng Ma
Date: 2022-02-18 15:25:59
LastEditTime: 2022-03-29 15:02:23
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/scripts/prepare_data.py

'''

import argparse
import os
import pandas as pd
import numpy as np

import h5py
import json

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

def main(args):

    # load csv file
    pfas_df = pd.read_csv(args.pfas_csv)   # columns = 'smiles', 'tokens', 'formula', 'precursor_mz', 'mz', 'intensity'
    print('total data num', len(pfas_df))
    # pfas_df_high = pd.read_csv('./data/cfmid3_0/high.csv')
    # pfas_df_medium = pd.read_csv('./data/cfmid3_0/medium.csv')
    # pfas_df_low = pd.read_csv('./data/cfmid3_0/low.csv') 
    

    # mol_mass represents molecular mass
    mz_count, token_count, mol_mass_count, formula_count = {}, {}, {}, {}
    mz_all, tokens_all, intensity_all, precursor_all, mol_mass_all, formula_all, = [], [], [], [], [], []
    smiles_dict = {}
    
    # load split infomation
    with open(args.split_json, 'r') as f:
        infos = json.load(f)

    # # if input mix data
    # new_train_info = []
    # new_val_info = []
    # new_test_info = []
    # for idx in infos['train']:
    #     new_train_info.extend([idx*3, idx*3+1, idx*3+2])
    # for idx in infos['val']:
    #     new_val_info.extend([idx*3, idx*3+1, idx*3+2])
    # for idx in infos['test']:
    #     new_test_info.extend([idx*3, idx*3+1, idx*3+2])
    # infos['train'] = new_train_info
    # infos['val'] = new_val_info
    # infos['test'] = new_test_info
    # print(train_num, len(new_train_info))
    # print(val_num, len(new_val_info))
    # print(test_num, len(new_test_info))



    for item in pfas_df.itertuples():
        index = item[0]
        smiles = item.smiles
        tokens = item.tokens.split()
        formula = item.formula
        precursor = round(item.precursor_mz, args.precise)
        mz = [round(float(m), args.precise) for m in item.mz.split()]
        intensity = [float(i) for i in item.intensity.split()]

        mol_mass = round(precursor + atom2mass['H'] - atom2mass['E'], args.precise) # compute molecular mass


        mz_all.append(mz)
        tokens_all.append(tokens)
        intensity_all.append(intensity)
        precursor_all.append(precursor)
        mol_mass_all.append(mol_mass)
        formula_all.append(formula)
        smiles_dict[index] = smiles

        for m in mz:
            mz_count[m] = mz_count.get(m, 0) + 1
        mz_count[precursor] = mz_count.get(precursor, 0) + 1

        for token in tokens:
            token_count[token] = token_count.get(token, 0) + 1 

        mol_mass_count[mol_mass] = mol_mass_count.get(mol_mass, 0) + 1
        formula_count[formula] = formula_count.get(formula, 0) + 1

    # build vocabulary
    mz_vocab = [m for m,n in mz_count.items() if n >= args.mz_threshold]
    token_vocab = [token for token, n in token_count.items() if n >= args.token_threshold]
    mol_mass_vocab = [m for m,n in mol_mass_count.items() if n >= args.fm_threshold]
    formula_vocab = [m for m,n in formula_count.items() if n >= args.formula_threshold] 
    print('mz vocab size:', len(mz_vocab))
    print('token vocab size:', len(token_vocab))
    print('mm vocab size:', len(mol_mass_vocab))
    print('formula vocab size:', len(formula_vocab))


    # add UNK to represent mzs(or tokens) less than threshold
    mz_vocab.append('UNK')
    token_vocab.append('UNK')
    mol_mass_vocab.append('UNK')
    formula_vocab.append('UNK')

    ix_to_mz = {i+1:w for i,w in enumerate(mz_vocab)} # a 1-indexed vocab translation table, 0 for <pad>
    mz_to_ix = {w:i+1 for i,w in enumerate(mz_vocab)} # inverse table


    ix_to_token =  {i+1:w for i,w in enumerate(token_vocab)} # same as above
    token_to_ix = {w:i+1 for i,w in enumerate(token_vocab)} 

    ix_to_mol_mass =  {i:w for i,w in enumerate(mol_mass_vocab)} # a 0-indexed vocab translation table, do not need <pad>
    mol_mass_to_ix = {w:i for i,w in enumerate(mol_mass_vocab)} 

    ix_to_formula =  {i:w for i,w in enumerate(formula_vocab)} # a 0-indexed vocab translation table, do not need <pad>
    formula_to_ix = {w:i for i,w in enumerate(formula_vocab)} 


    # encode mz, token, mol_mass, formula
    mz_arrays, tokens_arrays, intensity_arrays, mol_mass_arrays, formula_arrays = [], [], [], [], []
    for i in range(len(mz_all)):
        mz, tokens, intensity = mz_all[i], tokens_all[i], intensity_all[i]
        precursor, mol_mass, formula = precursor_all[i], mol_mass_all[i], formula_all[i]
        
        # merge duplicated mz
        tmp_dict = {}
        for j in range(len(mz)):
            tmp_dict[mz[j]] = tmp_dict.get(mz[j], 0) + intensity[j] / sum(intensity)
        
        # sort by intensity
        mz = np.array([m for m, n in tmp_dict.items()])
        intensity = np.array([n for m, n in tmp_dict.items()])
        sorted_id = intensity.argsort()[::-1]
        mz = list(mz[sorted_id])
        intensity = list(intensity[sorted_id])
        
        # add precursor
        mz.insert(0, precursor)
        intensity.insert(0, 1)

        # encode mz and tokens
        Mi = np.zeros((1,args.mz_length), dtype='int')
        Ti = np.zeros((1,args.token_length), dtype='int')
        Ii = np.zeros((1,args.mz_length), dtype='float')

        for k,w in enumerate(mz):
            if k < args.mz_length:
                Mi[0, k] = mz_to_ix.get(w, mz_to_ix['UNK'])
                Ii[0, k] = intensity[k]
        
        for k,w in enumerate(tokens):
            if k < args.token_length:
                Ti[0, k] = token_to_ix.get(w, token_to_ix['UNK'])

        mz_arrays.append(Mi)
        tokens_arrays.append(Ti)
        intensity_arrays.append(Ii)
        mol_mass_arrays.append(mol_mass_to_ix.get(mol_mass, mol_mass_to_ix['UNK']))
        formula_arrays.append(formula_to_ix.get(formula, formula_to_ix['UNK']))
    
    mz_arrays = np.concatenate(mz_arrays, axis=0)
    tokens_arrays = np.concatenate(tokens_arrays, axis=0)
    intensity_arrays = np.concatenate(intensity_arrays, axis=0)
    mol_mass_arrays = np.array(mol_mass_arrays)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    h5_path = os.path.join(args.output_dir, 'data.h5')
    h5_file = h5py.File(h5_path, 'w')
    h5_file.create_dataset('mz', dtype='int', data=mz_arrays)
    h5_file.create_dataset('tokens', dtype='int', data=tokens_arrays)
    h5_file.create_dataset('intensity', dtype='float', data=intensity_arrays)
    h5_file.create_dataset('mol_mass', dtype='int', data=mol_mass_arrays)
    h5_file.create_dataset('formula', dtype='int', data=formula_arrays)

    # save vocab and other infomations
    json_path = os.path.join(args.output_dir, 'data.json')
    json_file = {'ix_to_mz': ix_to_mz, 'mz_to_ix': mz_to_ix, \
                'ix_to_token': ix_to_token, 'token_to_ix': token_to_ix, \
                'mol_mass_to_ix': mol_mass_to_ix, 'ix_to_mol_mass': ix_to_mol_mass, \
                'formula_to_ix': formula_to_ix, 'ix_to_formula': ix_to_formula, \
                'data': infos, 'gts': smiles_dict
                }
    with open(json_path, 'w') as f:
        json.dump(json_file, f)
    
    print(f'h5 and json files are saved to {args.output_dir}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pfas_csv', default='./data/lipid/high.csv')
    parser.add_argument('--split_json', default='./data/lipid/split_info.json')
    parser.add_argument('--output_dir', default='./data/lipid/high_p0', help='build two files(json and h5)')

    parser.add_argument('--mz_length', type=int, default=20)
    parser.add_argument('--token_length', type=int, default=100)

    parser.add_argument('--mz_threshold', type=int, default=5)
    parser.add_argument('--token_threshold', type=int, default=5)
    parser.add_argument('--fm_threshold', type=int, default=5)
    parser.add_argument('--formula_threshold', type=int, default=5)

    parser.add_argument('--precise', type=int, default=0, help="the precise of spectral, can be chosen from 0,1,2 ...")

    args = parser.parse_args()
    main(args)




        









        
        











