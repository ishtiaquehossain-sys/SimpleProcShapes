import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from paramvectordef import ParamVectorDef
from model import MeshModel
from dataset import get_data_loaders


def evaluate_scalar(y, y_pred):
    diff = np.abs(y_pred - y)
    diff = np.mean(diff, axis=0)
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print(diff, end='')
    print()


def evaluate_discrete(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=1, average=None)
    recall = recall_score(y, y_pred, zero_division=1, average=None)
    f1 = f1_score(y, y_pred, zero_division=1, average=None)
    scores = np.array([accuracy, precision.mean(), recall.mean(), f1.mean()])
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(scores, end='')
    print()


def print_report(version, predictions, targets):
    param_def = ParamVectorDef(version)
    for i, param in enumerate(param_def.params):
        y = targets[i]
        y_pred = predictions[i]
        if param.paramtype == 'scalar':
            evaluate_scalar(y, y_pred)
        elif param.paramtype == 'type':
            evaluate_discrete(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
        else:
            evaluate_discrete(y, y_pred)


def main(args):
    version = args.version
    data_dir = os.path.join('data', 'test')
    model_dir = os.path.join('models')
    batch_size = args.batch_size

    test_loader, _ = get_data_loaders(data_dir, version, batch_size=batch_size, split_ratio=1.0)
    model = MeshModel(version)
    model.load_state_dict(torch.load(os.path.join(model_dir, '{}.pt'.format(version))))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    it = iter(test_loader)
    predictions = []
    targets = []
    for i, batch in enumerate(tqdm(it, file=sys.stdout)):
        if torch.cuda.is_available():
            batch['meshes'] = batch['meshes'].cuda()
        model.eval()
        predictions.append(model(batch['meshes'], predict=True))
        targets.append(batch['vectors'])
    predictions = [np.vstack([predictions[x][y].numpy() for x in range(len(predictions))]) for y in range(len(predictions[0]))]
    targets = [np.vstack([targets[x][y].numpy() for x in range(len(targets))]) for y in range(len(targets[0]))]
    print_report(version, predictions, targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=5, help="PML version (1-5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parsed_args = parser.parse_args()
    main(parsed_args)
