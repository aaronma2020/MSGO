# MSGO

This repository provieds data and methods in the paper: \
**Pseudodata-based molecular structure generator to reveal unknown chemicals (under review)**

Authors: Nanyang Yu, Zheng Ma, Qi Shao, Laihui Li, Xuebing Wang, and Si Wei*

### Setup

### Environment
Python: 3.7 \
Torch: 1.7.1

### Data
For Training, we use 30k+ pseudo smiles-specturm pairs generated by cfmid (you can download the raw smiles lists file [here](https://www.aliyundrive.com/s/BXKd1ThQy19)). For evaluation, we use 300+ real specturm to verify our method (download [here](https://www.aliyundrive.com/s/JTVRbipqXLh)). 

### Model weights
We provide the MSGO model (donwloade here) trained use pseudo smiles-specturm pairs with whole methods mentioned in paper. you also can train you own model with other methods.


### Evaluation

Download the model weights in your path and  
```
python tools/eval.py --model_weights your_model_weights_dir
```

### Predict real data
We provide a example data in data/example.csv, and run:
```
python tools/eval_standard.py --log_path your_model_weights_dir --real_csv ./data/example --out_csv ./results.csv
```
---
## Todos
- [x] Release model weights
- [x] Release pseudo and real data
- [ ] Release training process