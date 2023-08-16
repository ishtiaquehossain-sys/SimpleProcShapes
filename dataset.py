import io
import os
import sys
import zlib
import time
import tarfile
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from paramvectordef import ParamVectorDef
from procedure import TableProcedure


def collate_batched_meshes(batch: List[Dict]):
    collated_dict = dict()
    verts = [x['verts'] for x in batch]
    faces = [x['faces'] for x in batch]
    textures = [x['texture'] for x in batch]
    textures = TexturesVertex(verts_features=textures)
    collated_dict['meshes'] = Meshes(verts=verts, faces=faces, textures=textures)
    collated_dict['vectors'] = [torch.stack([batch[i]['vector'][j] for i in range(len(batch))]) for j in range(len(batch[0]['vector']))]
    return collated_dict


class MeshDataset(Dataset):
    def __init__(self, data_file):
        t_file = tarfile.open(data_file, 'r|')
        tar_content = dict()
        t_info = t_file.next()
        while t_info is not None:
            file_obj = t_file.extractfile(t_info)
            buf = zlib.decompress(file_obj.read())
            arr = np.load(io.BytesIO(buf))
            tar_content[t_info.name] = arr
            t_info = t_file.next()
        self.num_samples = len([x for x in tar_content.keys() if x.startswith('meshes')]) // 2
        self.vector_size = len([x for x in tar_content.keys() if x.startswith('vectors')])
        vertices = []
        faces = []
        textures = []
        for i in tqdm(range(self.num_samples), file=sys.stdout, desc='Reading meshes'):
            v = torch.from_numpy(tar_content['meshes/{}v.npy.z'.format(i)].astype(np.float32))
            f = torch.from_numpy(tar_content['meshes/{}f.npy.z'.format(i)])
            v_rgb = torch.ones_like(v)
            vertices.append(v)
            faces.append(f)
            textures.append(v_rgb)
        textures = TexturesVertex(verts_features=textures)
        self.meshes = Meshes(verts=vertices, faces=faces, textures=textures)
        self.vectors = [torch.from_numpy(tar_content['vectors/{}.npy.z'.format(i)].astype(np.float32)) for i in range(self.vector_size)]
        t_file.close()

    def __getitem__(self, index):
        verts, faces = self.meshes.get_mesh_verts_faces(index)
        texture = torch.squeeze(torch.stack(self.meshes.textures[index].verts_features_list()))
        datum = {
            'verts': verts,
            'faces': faces,
            'texture': texture,
            'vector': [self.vectors[i][index] for i in range(self.vector_size)]
        }
        return datum

    def __len__(self):
        return self.num_samples

    def close(self):
        self.meshes = None


def get_data_loaders(data_dir, version, batch_size=32, split_ratio=0.9, shuffle=True):
    dataset = MeshDataset(os.path.join(data_dir, '{}.tar'.format(version)))
    indices = list(range(len(dataset)))
    split_point = int(np.floor(split_ratio * len(dataset)))
    train_indices, test_indices = indices[:split_point], indices[split_point:]
    if shuffle:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
    else:
        train_sampler = SequentialSampler(train_indices)
        test_sampler = SequentialSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0,
                                collate_fn=collate_batched_meshes)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0,
                                collate_fn=collate_batched_meshes)
    return train_loader, test_loader


class NpyTarWriter(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'w|')

    def add(self, arr, prefix, name):
        sio = io.BytesIO()
        np.save(sio, arr)
        zbuf = zlib.compress(sio.getvalue())
        sio.close()

        tinfo = tarfile.TarInfo('{}{}{}'.format(prefix, name, '.npy.z'))
        tinfo.size = len(zbuf)
        tinfo.mtime = time.time()
        zsio = io.BytesIO(zbuf)
        zsio.seek(0)
        self.tfile.addfile(tinfo, zsio)
        zsio.close()

    def close(self):
        self.tfile.close()


def get_indices(dataset_size, split_ratio):
    indices = list(range(dataset_size))
    split_point = int(np.floor(split_ratio * dataset_size))
    return indices[:split_point], indices[split_point:]


def make_dataset(args):
    version = args.version
    param_def = ParamVectorDef(version)
    param_vectors = param_def.get_random_vectors(max_len=5000)
    proc = TableProcedure()
    meshes = proc.get_meshes(version, param_vectors)
    param_vectors = param_def.encode(param_vectors)
    train_indices, test_indices = get_indices(len(meshes), 0.9)
    tar_writer = NpyTarWriter(os.path.join('data', 'train', '{}.tar'.format(version)))
    for i, idx in enumerate(tqdm(train_indices, file=sys.stdout, desc='Writing training data')):
        tar_writer.add(meshes[idx].vertices, 'meshes/', '{}v'.format(i))
        tar_writer.add(meshes[idx].faces, 'meshes/', '{}f'.format(i))
    for i, subvector in enumerate(param_vectors):
        tar_writer.add(subvector[train_indices], 'vectors/', '{}'.format(i))
    tar_writer.close()
    tar_writer = NpyTarWriter(os.path.join('data', 'test', '{}.tar'.format(version)))
    for i, idx in enumerate(tqdm(test_indices, file=sys.stdout, desc='Writing test data')):
        tar_writer.add(meshes[idx].vertices, 'meshes/', '{}v'.format(i))
        tar_writer.add(meshes[idx].faces, 'meshes/', '{}f'.format(i))
    for i, subvector in enumerate(param_vectors):
        tar_writer.add(subvector[test_indices], 'vectors/', '{}'.format(i))
    tar_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=5, help="PML version (1-5)")
    parsed_args = parser.parse_args()
    make_dataset(parsed_args)
