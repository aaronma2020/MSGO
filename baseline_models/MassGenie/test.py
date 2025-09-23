import json
import pickle
from train import *
from exp import *

torch.set_grad_enabled(False)


def apply_dropout(m):
    if type(m) is nn.Dropout:
        m.train()


def token_to_smi(token):
    return ''.join(token)


def generate_candidates(loader, model, device, idx_to_token):
    all_tgt_y_list = []
    candidate_list = []
    for k in tqdm(range(500)):
        all_tgt_list = []
        for src, tgt, tgt_y in loader:
            src_key_padding_mask = get_key_padding_mask(src, device)
            src = src.cuda(device)
            tgt_mask = generate_square_subsequent_mask(tgt.size()[0], tgt.size()[-1], device)
            for i in range(99):
                tgt_key_padding_mask = get_key_padding_mask(tgt, device)
                tgt = tgt.cuda(device)
                out = model(src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
                predict = model.module.predictor(out[:, i])
                y = torch.argmax(predict, dim=1)

                if i < 98:
                    for idx, x in enumerate(tgt):
                        x[i+1] = y[idx]
                else:
                    tgt = torch.cat([tgt, y.unsqueeze(1)], dim=1)

            tgt_list = []
            for idx in tgt.tolist():
                tgt_list.append([idx_to_token[str(i)] for i in idx if i > 2])
            tgt_list = [token_to_smi(i) for i in tgt_list]
            all_tgt_list.extend(tgt_list)
            if k == 0:
                tgt_y_list = []
                for idx in tgt_y.tolist():
                    tgt_y_list.append([idx_to_token[str(i)] for i in idx if i > 2])
                tgt_y_list = [token_to_smi(i) for i in tgt_y_list]
                all_tgt_y_list.extend(tgt_y_list)
        candidate_list.append(all_tgt_list)

    return candidate_list, all_tgt_y_list


def test(val_loader, test_loader, exp_loader, model, device, idx_to_token):
    d_model = 1024
    model.eval()
    model.apply(apply_dropout)

    # val_candidates, val_ground_truth = generate_candidates(val_loader, model, device, idx_to_token)
    # test_candidates, test_ground_truth = generate_candidates(test_loader, model, device, idx_to_token)
    exp_candidates, exp_ground_truth = generate_candidates(exp_loader, model, device, idx_to_token)

    # val_dict = {}
    # test_dict = {}
    # for idx, ground_truth in enumerate(test_ground_truth):
    #     sample_dict = {'ground_truth': ground_truth, 'candidates': [i[idx] for i in test_candidates]}
    #     test_dict[str(idx)] = sample_dict
    exp_dict = {}
    for idx, ground_truth in enumerate(exp_ground_truth):
        sample_dict = {'ground_truth': ground_truth, 'candidates': [i[idx] for i in exp_candidates]}
        exp_dict[str(idx)] = sample_dict

    with open("lipid_test_result/exp_result_dict.pkl", 'wb') as f:
        pickle.dump(exp_dict, f)

    return


def main():
    file_path = "模型复现/lipid/MassGenie/data_dict.pkl"
    exp_path = "模型复现/lipid/MassGenie/exp_dict.pkl"
    data_dict = load_data(file_path)
    exp_dict = load_data(exp_path)

    idx_to_token = data_dict['idx_to_token']

    train_src, train_tgt, train_tgt_y = get_data(data_dict, 'train')
    val_src, val_tgt, val_tgt_y = get_data(data_dict, 'val')
    test_src, test_tgt, test_tgt_y = get_data(data_dict, 'test')
    exp_src, exp_tgt, exp_tgt_y = get_data_from_exp(exp_dict)

    for x in val_tgt:
        x[1:] = 0
    for x in test_tgt:
        x[1:] = 0
    for x in exp_tgt:
        x[1:] = 0

    train_set = TransformerDataset(train_src, train_tgt, train_tgt_y)
    val_set = TransformerDataset(val_src, val_tgt, val_tgt_y)
    test_set = TransformerDataset(test_src, test_tgt, test_tgt_y)
    exp_set = TransformerDataset(exp_src, exp_tgt, exp_tgt_y)

    batch_size = 32

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    exp_loader = DataLoader(exp_set, batch_size=batch_size, shuffle=False)

    device_ids = [0, 1]
    device = device_ids[0]

    model = TransformerModel(d_model=1024,
                             src_dim=50000,
                             tgt_dim=30,
                             )
    model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device)

    state_dict = torch.load("lipid_models/862.pt")
    model.load_state_dict(state_dict)

    model = model.cuda(device)

    model = test(val_loader, test_loader, exp_loader, model, device, idx_to_token)


if __name__ == '__main__':
    main()




