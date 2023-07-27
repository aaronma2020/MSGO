'''
Author: Zheng Ma
Date: 2022-02-19 15:39:15
LastEditTime: 2022-05-27 14:49:54
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/opt.py

'''

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--id', type=str, default='precursor_formula_mol_mass')
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--flag_metric', type=str, default='accurate')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--beam_size', type=int, default=1)

# model variant
parser.add_argument('--use_precursor', type=int, default=1)
parser.add_argument('--use_mask', type=int, default=0)
parser.add_argument('--use_formula', type=int, default=1)
parser.add_argument('--use_mol_mass', type=int, default=1)

# load data
parser.add_argument('--input_json', type=str, \
    default='./data/lipid/high_p0/data.json')
parser.add_argument('--input_label_h5', type=str, \
    default='./data/lipid/high_p0/data.h5')

# save params
parser.add_argument('--log', type=str, default='./log/lipid/high_p0/')
parser.add_argument('--loss_save_step', type=int, default=25)

# model params
parser.add_argument('--N_enc', type=int, default=6)
parser.add_argument('--N_dec', type=int, default=6)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--num_att_heads', type=int, default=8)
parser.add_argument('--dropot', type=float, default=0.1)

# vocabulary setting
parser.add_argument('--bos_idx', type=int, default=0)
parser.add_argument('--eos_idx', type=int, default=0)
parser.add_argument('--pad_idx', type=int, default=0)


# optimizer params
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--learning_rate', type=float, default=4e-4)
parser.add_argument('--optim_alpha', type=float, default=0.9, \
    help='alpha for adam')
parser.add_argument('--optim_beta', type=float, default=0.999, \
    help='beta used for adam')
parser.add_argument('--optim_epsilon', type=float, default=1e-8, \
    help='epsilon that goes into denominator for smoothing')
parser.add_argument('--weight_decay', type=float, default=0, \
    help='weight_decay')



args = parser.parse_args()
if not os.path.exists(args.log):
    os.makedirs(args.log)

args.checkpoint_path = args.log +  args.id
if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)


