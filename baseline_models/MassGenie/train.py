import os
import sys
import queue
import torch.nn as nn
import numpy as np
from dataset import *
from model import TransformerModel
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

torch.manual_seed(42)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)

    return data_dict


def get_key_padding_mask(tokens, device):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == 0] = -torch.inf

    return key_padding_mask.cuda(device)


def generate_square_subsequent_mask(batch_size, sz, device):
    nb_heads = 16
    x = torch.triu(
        torch.full((sz, sz), float('-inf')),
        diagonal=1,
        )
    x = x.unsqueeze(0)
    x = x.repeat(batch_size*nb_heads, 1, 1)
    return x.cuda(device)


def adjust_learning_rate(optimizer, epoch, d_model, warm_steps):
    lr = d_model**(-0.5) * min(epoch**(-0.5), epoch*warm_steps**(-1.5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(loader, model, criteria, device):
    errors = []
    model.eval()
    with torch.no_grad():
        for src, tgt, tgt_y in loader:
            n_tokens = (tgt_y != 0).sum()
            src_key_padding_mask = get_key_padding_mask(src, device)
            tgt_key_padding_mask = get_key_padding_mask(tgt, device)
            tgt_mask = generate_square_subsequent_mask(tgt.size()[0], tgt.size()[-1], device)
            src = src.cuda(device)
            tgt = tgt.cuda(device)
            tgt_y = tgt_y.cuda(device)

            out = model(src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
            out = model.module.predictor(out)
            error = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
            errors.append(error.item())
    return np.mean(errors)


def early_stopping(losses):
    print(np.std(losses))
    if np.std(losses) < 0.0000001:
        sys.exit('Valid loss converging!')
    counter = 0
    if losses[4] > losses[0]:
        counter = counter + 1
    if losses[3] > losses[0]:
        counter = counter + 1
    if losses[2] > losses[0]:
        counter = counter + 1
    if losses[1] > losses[0]:
        counter = counter + 1
    if counter > 4:
        sys.exit('Loss increasing!')
    return


def train(train_loader, val_loader, model, epochs, device):
    d_model = 1024
    warm_steps = 8000
    start_step = 1
    start_lr = d_model**(-0.5) * min(start_step**(-0.5), start_step*warm_steps**(-1.5))
    optimizer = optim.Adam(model.parameters(), lr=start_lr, betas=(0.9, 0.98), eps=1e-9)
    criteria = nn.CrossEntropyLoss()
    model.train()
    epoch_losses = []

    valid_queue = queue.Queue(5)
    for epoch in tqdm(range(1, epochs)):
        adjust_learning_rate(optimizer, epoch, d_model, warm_steps)
        flag = 0
        for src, tgt, tgt_y in train_loader:
            if flag == 36:
                breakpoint()
            optimizer.zero_grad()
            n_tokens = (tgt_y != 0).sum()

            src_key_padding_mask = get_key_padding_mask(src, device)
            tgt_key_padding_mask = get_key_padding_mask(tgt, device)
            tgt_mask = generate_square_subsequent_mask(tgt.size()[0], tgt.size()[-1], device)
            src = src.cuda(device)
            tgt = tgt.cuda(device)
            tgt_y = tgt_y.cuda(device)

            out = model(src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
            out = model.module.predictor(out)

            loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            flag += 1

        print('Epoch: ', epoch, ', Train Loss: ', np.mean(epoch_losses))
        epoch_losses = []
        valid_loss = evaluate(val_loader, model, criteria, device)
        print('Epoch: ', epoch, ', Valid Loss: ', valid_loss)
        if epoch > 0 and epoch % 5 == 0:
            model_dir = os.path.join("models", str(epoch) + '.pt')
            torch.save(model.state_dict(), model_dir)
        if epoch > 5:
            rem = valid_queue.get()
        valid_queue.put(valid_loss)
        if epoch > 5:
            early_stopping(list(valid_queue.queue))

    return model


def main():
    file_path = "模型复现/lipid/MassGenie/data_dict.pkl"
    data_dict = load_data(file_path)

    train_src, train_tgt, train_tgt_y = get_data(data_dict, 'train')
    val_src, val_tgt, val_tgt_y = get_data(data_dict, 'val')
    test_src, test_tgt, test_tgt_y = get_data(data_dict, 'test')

    train_set = TransformerDataset(train_src, train_tgt, train_tgt_y)
    val_set = TransformerDataset(val_src, val_tgt, val_tgt_y)
    test_set = TransformerDataset(test_src, test_tgt, test_tgt_y)

    batch_size = 16

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

    epochs = 1000
    device_ids = [0, 1]
    device = device_ids[0]

    model = TransformerModel(d_model=1024,
                             src_dim=50000,
                             tgt_dim=30,
                             )
    model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device)

    model = model.cuda(device)

    model = train(train_loader, val_loader, model, epochs, device)


if __name__ == '__main__':
    main()


