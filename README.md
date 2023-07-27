# MSGO

This repository provieds data and methods in the paper: \
**Pseudodata-based molecular structure generator to reveal unknown chemicals (under review)**

Authors: Nanyang Yu, Zheng Ma, Qi Shao, Laihui Li, Xuebing Wang, and Si Wei*

### Setup

### Environment
Python: 3.7 \
Torch: 1.7.1

### Data
For Training, we use 30k+ pseudo smiles-specturm pairs generated by cfmid (you can download the raw smiles lists file [here](https://www.aliyundrive.com/s/BXKd1ThQy19)). For evaluation, we use 300+ real specturm to verify our method (download [here](https://www.aliyundrive.com/s/JTVRbipqXLh)). For evaluation in real samples，we use one LC–QTOF dataset for wastewater samples to verify our model (download [here](https://pan.baidu.com/s/1KuKzkRMGr1DhkfJobkXKpQ), code: gmas).

### Model weights
We provide the MSGO model ([pfas](https://pan.baidu.com/s/1J_qllzAsv-dxqD2D28_sLw), code: 0bfg; [lipid](https://pan.baidu.com/s/1lbL7hdHgblsQAgkBomlBMg), code: 37it) trained use pseudo smiles-specturm pairs with whole methods mentioned in paper. you also can train you own model with other methods.


### Evaluation

Download the model weights in ckpts/pfas or ckpts/lipid, run
```
python tools/eval.py --log_path [ckpt/pfas or ckpts/lipid]
```

### Predict real data
We provide example data in data/example.

For pfas, run :
```
python tools/eval_standard.py --log_path ckpts/pfas --real_csv ./data/example/pfas.csv --out_csv ./pfas_results.csv --beam_size 500 --polar neg
```

For lipid, run:
```
python tools/eval_standard.py --log_path ckpts/lipid --real_csv ./data/example/lipid.csv --out_csv ./lipid_results.csv --beam_size 300 --polar pos
```
Then you can obatin a results csv file inluding top 10 predicts.

---
## Todos
- [x] Release model weights
- [x] Release pseudo and real data
- [ ] Release training process