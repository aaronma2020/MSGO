import argparse
import pandas as pd
from transformers import AutoTokenizer, BartForConditionalGeneration

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import re
from tqdm import tqdm

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


def main(args):
    model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    testdata = pd.read_csv(args.real_csv)

    result = pd.DataFrame()
    for index, row in tqdm(testdata.iterrows()):
        # mz_list = row["Unnamed: 5"].split()
        mz_list = row["mz"].split()
        input_mz = []

        # 增加母离子信息

        # precusor_mz = str(round(float(row["precusor_mz"]), 1)).split(".")
        # input_mz.extend(precusor_mz)
        try:
            for mz in mz_list:
                mz = str(float(round(float(mz),1))).split(".")
                input_mz.extend(mz)
        except:
            continue
        input_mz = " ".join(input_mz)
        inputs = tokenizer([input_mz], max_length=1024, return_tensors="pt")
        output = model.generate(inputs["input_ids"], num_beams=args.beam_size, min_length=0, max_length=512, num_return_sequences=25)
        smiles_list = tokenizer.batch_decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        gt_smile = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"]))
        top10_smiles = []
        pattern = r'<\|spe\|>(.*?)</s>'
        for smiles in smiles_list:
            try:

                smiles = re.findall(pattern, smiles, re.DOTALL)[0]
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            except:
                smiles = smiles
            top10_smiles.append(smiles)

        for i, smiles in enumerate(top10_smiles):
            row[f"top{i+1}_smiles"] = smiles
            e,f = ecfp_fcfp(gt_smile, smiles)
            row[f"top{i+1}_ecfp"] = e
            row[f"top{i+1}_fcfp"] = f
            row[f"top{i+1}_acc"] = 1 if gt_smile == smiles else 0
        result = result.append(row, ignore_index=True)


    
    result.to_csv("./bart_result_lipid_bs25.csv")



        





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="./fine_tuned_bart_lipid")
    parser.add_argument("--real_csv", default="./data/real_lipid.csv")
    parser.add_argument("--beam_size", default=25)
    args = parser.parse_args()
    main(args)




