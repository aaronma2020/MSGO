'''
Author: Zheng Ma
Date: 2022-02-22 10:22:31
LastEditTime: 2022-05-04 10:46:56
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/tools/eval.py

'''

import torch
import argparse
import os
import sys
sys.path.append('.')
from dataloader import SmilesSet
from models.TransModel import TransModel
from utils.loader import build_loader
from utils.eval import eval

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', required=True, type=str)
parser.add_argument('--append', type=str, default='best')
parser.add_argument('--save_result', type=str, default='./results/')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--beam_size', type=int ,default=1)
eval_args = parser.parse_args()


# load checkpoint file
ckpt_path = os.path.join(eval_args.log_path, 'model'+eval_args.append+'.tar')
checkpoint = torch.load(ckpt_path)

# update opt
opt = checkpoint['opt']
opt.beam_size = eval_args.beam_size
opt.batch_size = eval_args.batch_size

# build dataloader
dataset = SmilesSet(opt)
test_loader = build_loader(dataset, 'test', opt.batch_size, False)

model = TransModel(opt).cuda()
model.load_state_dict(checkpoint['model'])

print('Testing')
results = eval(model, test_loader, opt)
for metric, score in results.items():
    print(metric, score)
