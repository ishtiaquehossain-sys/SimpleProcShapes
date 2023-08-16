import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import torch
from dataset import get_data_loaders
from model import MeshModel, MultiTaskLoss


def get_optimizer(model: MeshModel):
    camera_params = []
    feature_params = []
    scalar_params = []
    type_params = []
    binary_params = []
    for name, value in model.named_parameters():
        if 'camera' in name:
            camera_params.append(value)
        elif 'feature' in name:
            feature_params.append(value)
        elif 'scalar' in name:
            scalar_params.append(value)
        elif 'type' in name:
            type_params.append(value)
        elif 'binary' in name:
            binary_params.append(value)
    param_dict_list = list()
    param_dict_list.append({'params': camera_params, 'lr': 0.0005, 'weight_decay': 1e-8})
    param_dict_list.append({'params': feature_params, 'lr': 1e-5, 'weight_decay': 1e-8})
    param_dict_list.append({'params': scalar_params, 'lr': 1e-5, 'weight_decay': 1e-8})
    param_dict_list.append({'params': type_params, 'lr': 1e-5, 'weight_decay': 1e-8})
    param_dict_list.append({'params': binary_params, 'lr': 1e-5, 'weight_decay': 1e-8})
    optim = torch.optim.Adam(param_dict_list)
    return optim


def run_epoch(model: MeshModel, loss: MultiTaskLoss, data_loader, train=False, optim=None):
    epoch_loss = 0.0
    task_losses = []
    it = iter(data_loader)
    for i, batch in enumerate(tqdm(it, file=sys.stdout)):
        if torch.cuda.is_available():
            batch['meshes'] = batch['meshes'].cuda()
            batch['vectors'] = [x.cuda() for x in batch['vectors']]
        if train:
            model.train()
            optim.zero_grad()
            predictions = model(batch['meshes'])
            losses, combined_loss = loss(predictions, batch['vectors'])
            combined_loss.backward()
            optim.step()
        else:
            model.eval()
            predictions = model(batch['meshes'])
            losses, combined_loss = loss(predictions, batch['vectors'])
        epoch_loss += combined_loss.item()
        task_losses.append(losses.detach().cpu().numpy())
    epoch_loss /= len(data_loader.sampler)
    task_losses = np.array(task_losses).sum(axis=0) / len(data_loader.sampler)
    return epoch_loss, task_losses


def main(args):
    version = args.version
    data_dir = os.path.join('data', 'train')
    model_dir = os.path.join('models')
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    train_loader, valid_loader = get_data_loaders(data_dir, version, batch_size=batch_size)
    model = MeshModel(version)
    loss = MultiTaskLoss(version)
    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()
    optim = get_optimizer(model)

    for e in range(num_epochs):
        train_loss, task_losses = run_epoch(model, loss, train_loader, True, optim)
        valid_loss, task_losses = run_epoch(model, loss, valid_loader)
        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        print(f'Epoch {e + 1}: T Loss: {train_loss}, V Loss: {valid_loss}')
        print(task_losses, end='')
        print()

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, '{}.pt'.format(version)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=5, help="PML version (1-5)")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train validation split ratio")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parsed_args = parser.parse_args()
    main(parsed_args)
