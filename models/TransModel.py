'''
Author: Zheng Ma
Date: 2022-02-19 15:14:24
LastEditTime: 2022-03-31 10:38:21
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/models/TransModel.py

'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import copy

from .transformer import *

class TransModel(nn.Module):

    def make_model(self, mz_size, token_size, mol_mass_size, formula_size, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        
        # model = EncoderDecoder(
        #     Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
        #     Decoder(DecoderLayer(d_model, c(attn), c(attn), 
        #                         c(ff), dropout), N_dec),
        #     Embeddings(d_model, mz_size), # lambda x:x
        #     nn.Sequential(Embeddings(d_model, token_size), c(position)),
        #     Generator(d_model, token_size))

        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                c(ff), dropout), N_dec),
            Embeddings(d_model, mz_size), # lambda x:x
            DecodeEmbedding(d_model, token_size, formula_size, mol_mass_size, dropout),
            Generator(d_model, token_size))

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model  


    def __init__(self, opt):
        super(TransModel, self).__init__()

        self.mz_size = opt.mz_size + 1 # add pad
        self.token_size = opt.token_size + 1 # add pad
        if opt.use_mol_mass:
            self.mol_mass_size = opt.mol_mass_size
        else:
            self.mol_mass_size = None
        if opt.use_formula:
            self.formula_size = opt.formula_size
        else:
            self.formula_size = None
        self.token_length = opt.token_length

        self.N_enc = getattr(opt, 'N_enc', 6)
        self.N_dec = getattr(opt, 'N_dec', 6)
        self.d_model = getattr(opt, 'd_model', 512)
        self.d_ff = getattr(opt, 'd_ff', 2048)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        self.bos_idx = opt.bos_idx
        self.eos_idx = opt.eos_idx
        self.pad_idx = opt.pad_idx
        self.unk_idx = opt.token_to_ix['UNK']

        self.model = self.make_model(self.mz_size, self.token_size, self.mol_mass_size, self.formula_size, \
                                    self.N_enc, self.N_dec, self.d_model, self.d_ff, self.h, self.dropout
                                    )  

    def forward(self, mz, tokens, formula = None, mol_mass=None, mz_mask=None, tokens_mask=None):
        mz_mask, tokens_mask = self.prepare_mask(mz, tokens, mz_mask, tokens_mask)
        out = self.model(mz, tokens, formula, mol_mass, mz_mask, tokens_mask)
        outputs = self.model.generator(out)

        return outputs

    def prepare_mask(self, mz, tokens, mz_mask, tokens_mask):

        if mz_mask == None:
            mz_mask = mz.new_ones(mz.shape[:2], dtype=torch.long)
        mz_mask = mz_mask.unsqueeze(-2)

        if tokens_mask == None:
            tokens_mask = (tokens.data != self.eos_idx) & (tokens.data != self.pad_idx)
            tokens_mask[:,0] = 1 # bos
        
        tokens_mask = tokens_mask.unsqueeze(-2)
        tokens_mask = tokens_mask & subsequent_mask(tokens.size(-1)).to(tokens)
        return mz_mask, tokens_mask

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def sample_next_word(self, logprobs, sample_method):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
        return sampleLogprobs, it

        
    def sample(self, mz, mz_mask, formula, mol_mass, beam_size=1, device='cuda'):
        batch_size = mz.size(0)
        mz_mask = mz_mask.unsqueeze(-2)
        memory = self.model.encode(mz, mz_mask)
        seq = mz.new_full((batch_size, self.token_length + 1), self.pad_idx, dtype=torch.long)

        if beam_size == 1:
            for t in range(self.token_length + 1):
                if t == 0: # inptut <bos>
                    state = mz.new_full((batch_size, 1), self.bos_idx, dtype=torch.long)    # patial sequence beed predicted
                out = self.model.decode(memory, mz_mask, state, subsequent_mask(state.size(1)).to('cuda'), formula, mol_mass)

                logprobs = F.log_softmax(self.logit(out[:,-1]),dim=1)
                if self.unk_idx is not None: # not predict UNK
                    logprobs[:, self.unk_idx] = float('-inf')

                if t == self.token_length:
                    break

                sampleLogprobs, it = self.sample_next_word(logprobs, 'greedy')

                if t == 0:
                    unfinished = it != self.eos_idx
                else:
                    it[~unfinished] = self.pad_idx
                    logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                    unfinished = unfinished & (it != self.eos_idx)
                seq[:,t] = it
                if unfinished.sum() == 0:
                    break
                state = torch.cat([state, it.unsqueeze(1)], dim=1)

            return seq

        else:    # beam search

            def beam_step(logprobs, beam_size, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
                batch_size = beam_logprobs_sum.shape[0]
                vocab_size = logprobs.shape[-1]
                logprobs = logprobs.reshape(batch_size, -1, vocab_size)
                if self.unk_idx is not None: # not predict UNK
                    logprobs[:, :, self.unk_idx] = float('-inf')
                if t == 0:
                    beam_logprobs_sum = beam_logprobs_sum[:, :1]
                    state = repeat_tensors(beam_size, state)
                    
                
                candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs # beam_logprobs_sum Nxb logprobs is NxbxV
                ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
                ys, ix = ys[:,:beam_size], ix[:,:beam_size]
                beam_ix = ix // vocab_size # Nxb which beam
                selected_ix = ix % vocab_size # Nxb # which world
                state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1) # N*b which in Nxb beams

                if t > 0:
                    state = state.gather(0, state_ix.unsqueeze(-1).expand_as(state)) 
                    beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
                    beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(beam_seq_logprobs))

                beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1) # beam_seq Nxbxl
                beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + logprobs.reshape(batch_size, -1).gather(1, ix)


                beam_logprobs = logprobs.reshape(batch_size, -1, vocab_size).gather(1, beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size)) # NxbxV
                beam_seq_logprobs = torch.cat([beam_seq_logprobs, beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)
                
                state = torch.cat([state, selected_ix.reshape(-1).unsqueeze(-1)], -1)

                return beam_seq,beam_seq_logprobs, beam_logprobs_sum, state

            
            # init
            beam_seq_table = torch.LongTensor(batch_size, beam_size, 0).to(device)
            beam_seq_logprobs_table = torch.FloatTensor(batch_size, beam_size, 0, self.token_size).to(device)
            beam_logprobs_sum_table = torch.zeros(batch_size, beam_size).to(device)
            done_beams_table = [[] for _ in range(batch_size)]

            state = mz.new_full((batch_size, 1), self.bos_idx, dtype=torch.long) 
            out = self.model.decode(memory, mz_mask, state, subsequent_mask(state.size(1)).to('cuda'), formula, mol_mass)
            logprobs = F.log_softmax(self.logit(out[:,-1]),dim=1)

            memory = repeat_tensors(beam_size, memory)
            mz_mask = repeat_tensors(beam_size, mz_mask)
            formula = repeat_tensors(beam_size, formula) 
            mol_mass = repeat_tensors(beam_size, mol_mass)  
            for t in range(self.token_length+1):
                beam_seq_table, \
                beam_seq_logprobs_table,\
                beam_logprobs_sum_table, \
                state = beam_step(logprobs, \
                                beam_size, \
                                beam_seq_table, \
                                beam_seq_logprobs_table, \
                                beam_logprobs_sum_table, \
                                state)

                # if time's up... or if end token is reached then copy beams
                for b in range(batch_size):
                    is_end = beam_seq_table[b, :, t] == self.eos_idx
                    if t == self.token_length :
                        is_end.fill_(1)

                    for vix in range(beam_size):
                        if is_end[vix]:
                            final_beam = {
                                    'seq': beam_seq_table[b, vix].clone(), 
                                    'logps': beam_seq_logprobs_table[b, vix].clone(),
                                    'p': beam_logprobs_sum_table[b, vix].item()
                                    }
                            final_beam['p'] = final_beam['p'] # length penalty
                            done_beams_table[b].append(final_beam)
                    beam_logprobs_sum_table[b, is_end] -= 1000

                            

                # move the current group one step forward in time
                    
                out = self.model.decode(memory, mz_mask, state, subsequent_mask(state.size(1)).to('cuda'), formula, mol_mass) 
                logprobs = F.log_softmax(self.logit(out[:,-1]),dim=1)

            done_beams_table = [sorted(done_beams_table[b], key=lambda x: -x['p'])[:beam_size] for b in range(batch_size)]
            return done_beams_table




def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1) # Bx1x...
        x = x.expand(-1, n, *([-1]*len(x.shape[2:]))) # Bxnx...
        x = x.reshape(x.shape[0]*n, *x.shape[2:]) # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x









    
