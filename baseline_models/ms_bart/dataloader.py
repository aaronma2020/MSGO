import json
from random import randint, uniform
import h5py
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import os
from tqdm import tqdm
import random
from torch.utils.data import DataLoader

def read_csv(path):
    return pd.read_csv(path, names=["c1"])


class SmilesSet(data.Dataset):
    def __init__(self, tokenizer, type="pfas", split="all"):
        meta_data = {
            "pfas":{
                "formula": "./data/Formula.csv",
                "smiles": "./data/Smiles.csv",
                "specint": "./data/PFASSpecint.csv",
                "specfra": "./data/PFASSpecfra.csv",
                "new_token": "./data/PFAS_new_token.json",
                "infos": "./data/PFAS_infos.json"
            },
            "lipid":{
                "formula": "./data/LipidFormula.csv",
                "smiles": "./data/LipidSmiles.csv",
                "specint": "./data/LipidSpecint.csv",
                "specfra": "./data/LipidSpecfra.csv",
                "new_token": "./data/Lipid_new_token.json",
                "infos": "./data/Lipid_infos.json"
            }
        }
        meta_data = meta_data[type]
        if split in ["train", "val", "test"]:
            infos = json.load(open(meta_data["infos"]))["data"]
            self.formula = []
            self.smiles = []
            self.specint = []
            self.specfra = []
            formula = read_csv(meta_data["formula"])["c1"].to_list()
            smiles = read_csv(meta_data["smiles"])["c1"].to_list()
            specint = read_csv(meta_data["specint"])["c1"].to_list()
            specfra = read_csv(meta_data["specfra"])["c1"].to_list()
            print(len(formula))
            print(len(smiles))
            print(len(specint))
            print(len(specfra))
            data_ids = infos[split]
            for i in data_ids:
                self.formula.append(formula[i])
                self.smiles.append(smiles[i])
                self.specint.append(specint[i])
                self.specfra.append(specfra[i])

            # else:
            #     self.formula = read_csv(meta_data["formula"])["c1"].to_list()[infos["val"]]
            #     self.smiles = read_csv(meta_data["smiles"])["c1"].to_list()[infos["val"]]
            #     self.specint = read_csv(meta_data["specint"])["c1"].to_list()[infos["val"]]
            #     self.specfra = read_csv(meta_data["specfra"])["c1"].to_list()[infos["val"]]

        else:
            self.formula = read_csv(meta_data["formula"])["c1"].to_list()
            self.smiles = read_csv(meta_data["smiles"])["c1"].to_list()
            self.specint = read_csv(meta_data["specint"])["c1"].to_list()
            self.specfra = read_csv(meta_data["specfra"])["c1"].to_list()
        if not os.path.exists(meta_data["new_token"]):
            new_tokens = self.make_new_tokens()
            with open(meta_data["new_token"], "w") as f:
                json.dump(new_tokens, f)
        else:
            new_tokens = json.load(open(meta_data["new_token"]))
        self.mask_token = "<|mask|>"
        self.sep_token = "<|spe|>"
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        tokenizer.add_tokens([self.mask_token, self.sep_token], special_tokens=True)
        self.tokenizer = tokenizer
       

    def get_tokenizer(self):
        return self.tokenizer

    def make_new_tokens(self):
        tokens = []
        all_data = [self.formula, self.smiles, self.specint, self.specfra]
        print("处理tokens")
        for data in all_data:
            for item in tqdm(data):
                ts = item.split()
                for t in ts:
                    if t not in tokens:
                        tokens.append(t)
        
        return tokens
    
    def __getitem__(self, i):
        formula, smiles, specint, specfra = self.formula[i].split(), self.smiles[i].split(), self.specint[i].split(), self.specfra[i].split()
        # print(formula)
        # print(smiles)
        # print(len(specint))
        # print(len(specfra))
        inputs = []
        for j in range(len(specint)):
            inputs.extend([specint[j], specfra[j]])
        if random.randint(0, 1):
            inputs[:2] = [self.mask_token, self.mask_token]
        inputs = " ".join(inputs)
        inputs = self.tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
        labels = []
        labels.extend(formula)
        labels.append(self.sep_token)
        labels.extend(smiles)
        labels = " ".join(labels)
        labels = self.tokenizer(labels, max_length=256, truncation=True, padding="max_length")
        inputs["labels"] = labels['input_ids']
        return inputs

    def __len__(self):
        return len(self.formula)

    
if __name__ == "__main__":
    test = SmilesSet(None)