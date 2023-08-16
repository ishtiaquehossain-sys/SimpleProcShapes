import os
import sys
import csv
import random
from mathutils import Vector
import cv2 as cv
import argparse
import numpy as np
from tqdm import tqdm
import trimesh
import bpy
from sklearn.cluster import AgglomerativeClustering
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from paramvectordef import ParamVectorDef
from model import MeshModel
from procedure import TableProcedure
from lfd import LightFieldDistance


# Change the path to ShapeNet if necessary
shapenet_path = os.path.join('/', 'mnt', 'Research', 'ShapeNetCore.v2', '04379243')


def load_shapenet_obj(shape):
    file = os.path.join(shapenet_path, shape, 'models', 'model_normalized.obj')
    mesh = trimesh.load(file, force='mesh')
    mesh.apply_translation(-np.sum(mesh.bounding_box.bounds, axis=0) / 2)
    mesh.apply_scale(1 / np.max(mesh.extents))
    return mesh


def get_predictions(grnd_shapes, version, batch_size):
    v = [torch.from_numpy(np.array(x.vertices).astype(np.float32)) for x in grnd_shapes]
    f = [torch.from_numpy(np.array(x.faces)) for x in grnd_shapes]
    meshdata = Meshes(
        verts=v,
        faces=f,
        textures=TexturesVertex([torch.ones_like(x) for x in v])
    )
    model = MeshModel(version)
    model.load_state_dict(torch.load(os.path.join('models', '{}.pt'.format(version))))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    predictions = []
    for i in range(0, len(meshdata), batch_size):
        batch = meshdata[i:i + batch_size]
        if torch.cuda.is_available():
            batch = batch.cuda()
        model.eval()
        predictions.append(model(batch, predict=True))
    predictions = [np.vstack([predictions[x][y].numpy() for x in range(len(predictions))]) for y in
                   range(len(predictions[0]))]
    param_def = ParamVectorDef(version)
    predictions = param_def.decode(predictions)
    proc = TableProcedure()
    predicted_shapes = proc.get_meshes(version, predictions)
    return predicted_shapes


def get_lf_descriptors(shapes):
    num_shapes = len(shapes)
    features = []
    with tqdm(total=num_shapes, file=sys.stdout, desc='Calculating light field descriptors') as progress:
        for i in range(num_shapes):
            features.append(LightFieldDistance().get_descriptor(shapes[i].vertices, shapes[i].faces))
            progress.update()
    return features


def get_lf_distance_matrix(features):
    num_shapes = len(features)
    distances = np.zeros((num_shapes, num_shapes), dtype=int)
    for i in range(num_shapes):
        for j in range(num_shapes):
            distances[i, j] = LightFieldDistance().get_distance(features[i], features[j])
    return distances


def find_good_predictions(grnd_features, pred_features):
    distances = [LightFieldDistance().get_distance(gf, pf) for gf, pf in list(zip(grnd_features, pred_features))]
    distances = np.array(distances)
    threshold = 2700
    compatible_idx = np.where(distances < threshold)[0].tolist()
    incompatible_idx = np.where(distances >= threshold)[0].tolist()
    return compatible_idx, incompatible_idx


def get_clusters(distance_matrix, indices):
    cluster_result = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=3400,
        affinity='precomputed',
        linkage='complete'
    ).fit(distance_matrix[indices][:, indices])
    clusters = []
    for c in range(cluster_result.n_clusters_):
        clusters.append([indices[x] for x in np.where(cluster_result.labels_ == c)[0].tolist()])
    return clusters


def get_random_color():
    return random.random(), random.random(), random.random(), 1.0


def blender_init():
    bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)
    bpy.data.objects['Light'].data.use_shadow = False
    bpy.data.objects['Camera'].location = Vector((1.08, -1.46, 1.01))
    bpy.data.objects['Camera'].rotation_euler = Vector((1.05, 0.0, 0.65))
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'

    scene.use_nodes = True
    node_tree = scene.node_tree
    alpha_node = node_tree.nodes.new('CompositorNodeAlphaOver')
    alpha_node.premul = 1.0
    node_tree.links.new(node_tree.nodes['Render Layers'].outputs[0], alpha_node.inputs[2])
    node_tree.links.new(alpha_node.outputs[0], node_tree.nodes['Composite'].inputs[0])

    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = scene.render.resolution_y = 256


def render(verts, faces, color):
    for obj_name in bpy.data.objects.keys():
        if 'Cube' in obj_name or 'Cylinder' in obj_name:
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
    mesh = bpy.data.meshes.new('temp')
    v = verts[:, [0, 2, 1]]
    v[:, 1] = -v[:, 1]
    v = list(map(tuple, v))
    f = list(map(tuple, faces))
    mesh.from_pydata(v, [], f)
    mesh.update()
    obj = bpy.data.objects.new('temp', mesh)
    bpy.data.collections['Collection'].objects.link(obj)
    mat = bpy.data.materials.new("PKHG")
    mat.diffuse_color = color
    obj.active_material = mat
    bpy.context.scene.render.filepath = os.path.join('/', 'tmp', 'temp_vis.png')
    bpy.ops.render.render(write_still=True)
    img = cv.imread(bpy.context.scene.render.filepath)
    bpy.data.objects.remove(obj, do_unlink=True)
    return img


def main(args):
    version = args.version
    batch_size = args.batch_size

    blender_init()
    random.seed(30)

    f = open('shapenet.csv')
    reader = csv.reader(f)
    obj_list = [x[0] for x in reader]
    grnd_shapes = [load_shapenet_obj(x) for x in obj_list]
    grnd_features = get_lf_descriptors(grnd_shapes)
    lf_distance_matrix = get_lf_distance_matrix(grnd_features)
    if version > 0:
        pred_shapes = get_predictions(grnd_shapes, version=version, batch_size=batch_size)
        pred_features = get_lf_descriptors(pred_shapes)
        compatible_idx, incompatible_idx = find_good_predictions(grnd_features, pred_features)
    else:
        compatible_idx, incompatible_idx = [], [x for x in range(len(obj_list))]
    if len(incompatible_idx) > 0:
        clusters = get_clusters(lf_distance_matrix, incompatible_idx)
    else:
        clusters = []
    if len(compatible_idx) > 0:
        clusters.insert(0, compatible_idx)
    img = []
    for i, cluster in enumerate(clusters):
        if version > 0:
            grnd_color = (0.7, 0.7, 0.7, 1.0) if i == 0 else get_random_color()
            pred_color = (0.0, 1.0, 0.0, 1.0) if i == 0 else (1.0, 1.0, 0.0, 1.0)
            for idx in cluster:
                img.append(
                    np.vstack([
                        render(grnd_shapes[idx].vertices, grnd_shapes[idx].faces, grnd_color),
                        render(pred_shapes[idx].vertices, pred_shapes[idx].faces, pred_color)
                    ])
                )
        else:
            grnd_color = get_random_color()
            for idx in cluster:
                img.append(
                    render(grnd_shapes[idx].vertices, grnd_shapes[idx].faces, grnd_color)
                )
    col_len = 10
    img = [img[x:x+col_len] for x in range(0, len(img), col_len)]
    for i in range(col_len - len(img[-1])):
        if version > 0:
            img[-1].append(np.ones((512, 256, 3), dtype=np.uint8) * 255)
        else:
            img[-1].append(np.ones((256, 256, 3), dtype=np.uint8) * 255)
    img = np.vstack([np.hstack(x) for x in img])
    cv.imwrite(os.path.join('replicated_{}.png'.format(version)), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=5, help="PML version (0-5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parsed_args = parser.parse_args()
    main(parsed_args)
