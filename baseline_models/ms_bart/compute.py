import pandas as pd
import numpy as np

path = "./bart_result_25.csv"
test_data = pd.read_csv(path)


result = []
acc = []
ecfp = []
fcfp = []
for index, row in test_data.iterrows():
    tmp_acc = []
    tmp_ecfp = []
    tmp_fcfp = []
    for i in range(1, 26):
        tmp_acc.append(row[f"top{i}_acc"])
        tmp_ecfp.append(row[f"top{i}_ecfp"])
        tmp_fcfp.append(row[f"top{i}_fcfp"])

    tmp_sum_acc = []
    tmp_sum_ecfp = []
    tmp_sum_fcfp = []
    for i in range(1, 26):
        tmp_sum_acc.append(max(tmp_acc[:i]))
        tmp_sum_ecfp.append(max(tmp_ecfp[:i]))
        tmp_sum_fcfp.append(max(tmp_fcfp[:i]))

    acc.append(tmp_sum_acc)
    ecfp.append(tmp_sum_ecfp)
    fcfp.append(tmp_sum_fcfp)


for i in range(25):
    print(f"top{i+1}_acc:", np.mean([n[i] for n in acc]))
    print(f"top{i+1}_ecfp:", np.mean([n[i] for n in ecfp]))
    print(f"top{i+1}_fcfp:", np.mean([n[i] for n in fcfp]))
    

