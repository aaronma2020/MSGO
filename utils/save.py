'''
Author: Zheng Ma
Date: 2022-02-21 13:24:53
LastEditTime: 2022-02-21 13:24:53
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/utils/save.py

'''

import torch
import os

def save_model(model, opt, epoch, save_path, append=None):
    save_state = {}
    save_state['model'] = model.state_dict() 
    save_state['opt'] = opt
    save_state['epoch'] = epoch
    if append:
        save_path = os.path.join(save_path, 'model'+append+'.tar')
    else:
        save_path = os.path.join(save_path, f'model{epoch}.tar')

    torch.save(save_state, save_path)
