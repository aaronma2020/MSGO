'''
Author: Zheng Ma
Date: 2022-02-19 15:00:26
LastEditTime: 2022-02-28 11:19:50
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/dataloader.py

'''
from concurrent.futures import process
import json
from random import randint, uniform
import h5py
import torch

import numpy as np
import torch.utils.data as data

class SmilesSet(data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))

        # load vocabularies
        self.ix_to_mz = self.info['ix_to_mz']
        self.mz_size = len(self.ix_to_mz)
        print('mz size is ', self.mz_size)
        self.mz_to_ix = self.info['mz_to_ix']


        self.ix_to_token = self.info['ix_to_token']
        self.token_size = len(self.ix_to_token)
        print('token size is ', self.token_size)
        self.token_to_ix = self.info['token_to_ix']

        self.ix_to_mol_mass = self.info['ix_to_mol_mass']
        self.mol_mass_size = len(self.ix_to_mol_mass)
        print('formula weight size is ', self.mol_mass_size )
        self.mol_mass_to_ix = self.info['mol_mass_to_ix']

        self.ix_to_formula = self.info['ix_to_formula']
        self.formula_size = len(self.ix_to_formula) 
        print('formula size is ', self.formula_size)
        self.formula_to_ix = self.info['formula_to_ix']

        # load data
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
        mz_size = self.h5_label_file['mz'].shape
        self.mz = self.h5_label_file['mz'][:]
        self.mz_length = mz_size[1]
        print('max mz length in data is ', self.mz_length)

        token_size = self.h5_label_file['tokens'].shape
        self.token = self.h5_label_file['tokens'][:]
        self.token_length = token_size[1]
        print('max token length in data is ', self.token_length)
        
        self.intensity = self.h5_label_file['intensity']
        self.mol_mass = self.h5_label_file['mol_mass'][:]
        self.formula = self.h5_label_file['formula'][:]

        self.num_spectral = len(self.info['data'])
        self.split_ix = {'train': [], 'val': [], 'test': []}
        self.split_ix['train'] = self.info['data']['train']
        self.split_ix['val'] = self.info['data']['val']
        self.split_ix['test'] = self.info['data']['test']

        print('assigned %d data to split train' %len(self.split_ix['train']))
        print('assigned %d data to split val' %len(self.split_ix['val']))
        print('assigned %d data to split test' %len(self.split_ix['test']))


    def __getitem__(self, ix):

        # load mz, token, intensity, fomula,  molecular mass 
        token = self.token[ix]

            
        mz = self.mz[ix]
        intensity = self.intensity[ix]
        mol_mass = self.mol_mass[ix]
        formula = self.formula[ix]
        if self.opt.use_precursor == 0: # the first mz is precursor mz
            mz, intensity = mz[1:], intensity[1:]

        return mz, token, intensity, mol_mass, formula, ix

    def collate_func_train(self, batch): 

        mz_batch = []
        token_batch = []
        mol_mass_batch = []
        formula_batch = []
        
        infos = []
        for sample in batch:
            tmp_mz, tmp_token, tmp_intensity, tmp_mol_mass, tmp_formula, ix = sample
            if self.opt.use_mask:
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    tmp_mz = self.mask_process(tmp_mz, tmp_intensity)
            mz_batch.append(np.array(tmp_mz, dtype='int'))
            mol_mass_batch.append([tmp_mol_mass])
            formula_batch.append([tmp_formula])


            # add bos and eos
            tmp_label = np.zeros([self.token_length + 2], dtype = 'int')
            tmp_label[1 : self.token_length + 1] = tmp_token
            token_batch.append(tmp_label)
            tmp_gts =  self.info['gts'][str(ix)]

            # save other information
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['gts'] = tmp_gts
            infos.append(info_dict)


        data = {}
        data['mz'] = np.stack(mz_batch)
        data['mz_mask'] = np.zeros(data['mz'].shape[:2], dtype='float32')
        nonzeros = np.array(list(map(lambda x: (x != 0).sum(), data['mz'])))
        for i, l in enumerate(nonzeros):
            data['mz_mask'][i][:l] = 1

        data['token'] = np.stack(token_batch)
        data['token_mask'] = np.zeros(data['token'].shape[:2], dtype='int')
        data['mol_mass'] = np.stack(mol_mass_batch)
        data['formula'] = np.stack(formula_batch)

        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['token'])))
        for i, l in enumerate(nonzeros):
            data['token_mask'][i][:l] = 1
        data['info'] = infos

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data

    def collate_func_val(self, batch): 

        mz_batch = []
        token_batch = []
        mol_mass_batch = []
        formula_batch = []
        
        infos = []
        for sample in batch:
            tmp_mz, tmp_token, tmp_intensity, tmp_mol_mass, tmp_formula, ix = sample
            mz_batch.append(np.array(tmp_mz, dtype='int'))
            mol_mass_batch.append([tmp_mol_mass])
            formula_batch.append([tmp_formula])

            # add bos and eos
            tmp_label = np.zeros([self.token_length + 2], dtype = 'int')
            tmp_label[1 : self.token_length + 1] = tmp_token
            token_batch.append(tmp_label)
            tmp_gts =  self.info['gts'][str(ix)]

            # save other information
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['gts'] = tmp_gts
            infos.append(info_dict)


        data = {}
        data['mz'] = np.stack(mz_batch)
        data['mz_mask'] = np.zeros(data['mz'].shape[:2], dtype='float32')
        nonzeros = np.array(list(map(lambda x: (x != 0).sum(), data['mz'])))
        for i, l in enumerate(nonzeros):
            data['mz_mask'][i][:l] = 1

        data['token'] = np.stack(token_batch)
        data['token_mask'] = np.zeros(data['token'].shape[:2], dtype='int')
        data['mol_mass'] = np.stack(mol_mass_batch)
        data['formula'] = np.stack(formula_batch)

        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['token'])))
        for i, l in enumerate(nonzeros):
            data['token_mask'][i][:l] = 1
        data['info'] = infos

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data

    def mask_process(self, mz, intensity):

        num = (mz>0).sum()
        # intensity = intensity / intensity.sum()
        prob = np.random.uniform(0, 1, len(intensity))
        mask = intensity > prob

        if mask.sum() == 0 :
            return mz
        else:
            tmp_mz = [mz[i] for i in range(num) if mask[i]]
            new_mz = np.zeros_like(mz)
            for i, m in enumerate(tmp_mz):
                new_mz[i] = m
            return new_mz




    


        



