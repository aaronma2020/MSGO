'''
Author: Zheng Ma
Date: 2022-02-19 15:35:00
LastEditTime: 2022-02-28 12:18:58
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/tools/train.py

'''
import torch
import numpy as np
import random

# fix random seed
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataloader import SmilesSet
from models.TransModel import TransModel
from utils.loader import build_loader
from utils.optim import build_optimizer
from utils.loss import LanguageModelCriterion
from utils.eval import eval
from utils.save import save_model
from opt import args


def main(opt):

    # build dataloader
    dataset = SmilesSet(opt)
    train_loader = build_loader(dataset, 'train', opt.batch_size, True)
    val_loader = build_loader(dataset, 'val', opt.batch_size, False)

    opt.ix_to_mz = dataset.ix_to_mz
    opt.ix_to_token = dataset.ix_to_token
    opt.ix_to_mol_mass = dataset.ix_to_mol_mass
    opt.ix_to_formula = dataset.ix_to_formula

    opt.mz_to_ix = dataset.mz_to_ix
    opt.token_to_ix = dataset.token_to_ix
    opt.mol_mass_to_ix = dataset.mol_mass_to_ix
    opt.formula_to_ix = dataset.formula_to_ix

    opt.mz_size = dataset.mz_size
    opt.token_size = dataset.token_size
    opt.mol_mass_size = dataset.mol_mass_size
    opt.formula_size = dataset.formula_size

    opt.token_length = dataset.token_length

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    model = TransModel(opt).cuda()
    optimizer = build_optimizer(model.parameters(), opt)
    loss_func = LanguageModelCriterion()

    iteration = 1
    epoch = 1
    best_score = 0
    best_epoch = 0

    while True:
        # Training
        print(f'Training {epoch}/{opt.max_epoch} epochs')
        model.train()
        for i, data in tqdm(enumerate(train_loader)):
            tmp = [data['mz'], data['token'], data['token_mask'], data['mz_mask']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            mz, tokens, tokens_mask, mz_mask = tmp
            if opt.use_formula:
                formula = data['formula'].cuda()
            else:
                formula = None
            if opt.use_mol_mass:
                mol_mass = data['mol_mass'].cuda()
            else:
                mol_mass = None

            output = model(mz, tokens[..., :-1], formula, mol_mass, mz_mask, tokens_mask[..., :-1])
            loss = loss_func(output, tokens[..., 1:], tokens_mask[..., 1:]).mean()

            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()

            train_loss = loss.item()
            iteration += 1
            if iteration % opt.loss_save_step  == 0:
                tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
            # bar.dynamic_messages['loss'] = train_loss
            # bar.update(i)

        # Evaluating
        print('Testing')
        model.eval()
        results = eval(model, val_loader, opt)
        for metric, score in results.items():
            print(metric, score)
            tb_summary_writer.add_scalar(metric, score, epoch)

        epoch_score = results[opt.flag_metric]
        if epoch % 10 == 0:
            save_model(model, opt, epoch, opt.checkpoint_path)  
        if best_score < epoch_score:
            best_score = epoch_score
            best_epoch = epoch
            save_model(model, opt, epoch, opt.checkpoint_path, append='best')
        
        epoch += 1
        if epoch > opt.max_epoch:
            break 
        print(f'best {opt.flag_metric}: ', best_score)
        print(f'best epoch: ', best_epoch)
        print('-'*20)



if __name__ == '__main__':
    print('model variant params:')
    print('use_precursor:', args.use_precursor)
    print('use_mask:', args.use_mask)
    print('use_formula:', args.use_formula)
    print('user_mol_mass:', args.use_mol_mass)
    main(args)




