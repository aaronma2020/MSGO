'''
Author: Zheng Ma
Date: 2022-02-19 16:10:26
LastEditTime: 2022-02-19 16:10:40
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/utils/loader.py

'''
import torch

def build_loader(dataset, split, batch_size, shuffle, num_workers=4):
    sub_dataset = torch.utils.data.Subset(dataset, dataset.split_ix[split])
    if split == 'train':
        loader = torch.utils.data.DataLoader(
            dataset=sub_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_func_train
        ) 
    else:
        loader = torch.utils.data.DataLoader(
            dataset=sub_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_func_val
        ) 
        
    return loader


