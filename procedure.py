import sys
import argparse
from tqdm import tqdm
import numpy as np
import trimesh
import bpy
from paramvectordef import ParamVectorDef


def cube(loc=(0.0, 0.0, 0.0), scl=(1.0, 1.0, 1.0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, scale=scl)


def cylinder(loc=(0.0, 0.0, 0.0), scl=(1.0, 1.0, 1.0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=loc, scale=scl)


class TableProcedure:
    def create_table_v1(self, height, breadth, top_thickness, leg_thickness):
        h, b, t_th, l_th = height, breadth, height * top_thickness, breadth * leg_thickness
        # create table-top
        top_loc, top_scl = (0.0, 0.0, (h - t_th) / 2.0), (1.0, b, t_th)
        cube(loc=top_loc, scl=top_scl)

        leg_scl = (l_th, l_th, h)
        # create four legs
        x, y = (1.0 - l_th) / 2.0, (b - l_th) / 2.0
        leg_locs = [(x, y, 0.0), (x, -y, 0.0), (-x, y, 0.0), (-x, -y, 0.0)]
        for leg_loc in leg_locs:
            cube(loc=leg_loc, scl=leg_scl)

    def create_table_v2(self, height, breadth, top_thickness, leg_thickness, roundtop):
        h, b, t_th, l_th = height, breadth, height * top_thickness, breadth * leg_thickness
        rt = roundtop
        # create table-top
        top_loc, top_scl = (0.0, 0.0, (h - t_th) / 2.0), (1.0, b, t_th)
        cylinder(loc=top_loc, scl=top_scl) if rt else cube(loc=top_loc, scl=top_scl)

        leg_scl = (l_th, l_th, h)
        # create four legs
        a = 1.41 if rt else 1.0
        x, y = (1.0 - a * l_th) / (2 * a), (b - a * l_th) / (2 * a)
        leg_locs = [(x, y, 0.0), (x, -y, 0.0), (-x, y, 0.0), (-x, -y, 0.0)]
        for leg_loc in leg_locs:
            cube(loc=leg_loc, scl=leg_scl)

    def create_table_v3(self, height, breadth, top_thickness, leg_thickness, roundtop, leg_type):
        h, b, t_th, l_th  = height, breadth, height * top_thickness, breadth * leg_thickness
        rt, l_tp = roundtop, leg_type
        # create table-top
        top_loc, top_scl = (0.0, 0.0, (h - t_th) / 2.0), (1.0, b, t_th)
        cylinder(loc=top_loc, scl=top_scl) if rt else cube(loc=top_loc, scl=top_scl)

        leg_scl = (l_th, l_th, h)
        # create four legs
        a = 1.41 if rt else 1.0
        x, y = (1.0 - a * l_th) / (2 * a), (b - a * l_th) / (2 * a)
        leg_locs = [(x, y, 0.0), (x, -y, 0.0), (-x, y, 0.0), (-x, -y, 0.0)]
        for leg_loc in leg_locs:
            cube(loc=leg_loc, scl=leg_scl)
        # create support between legs
        if l_tp == 'support':
            leg_sprt_scl = (l_th, b / a, l_th) if rt else (l_th, b, l_th)
            cube(loc=(x, 0.0, -h / 3.0), scl=leg_sprt_scl)
            cube(loc=(-x, 0.0, -h / 3.0), scl=leg_sprt_scl)
            cube(loc=(0.0, 0.0, -h / 3.0), scl=((1.0 - a * l_th) / a, l_th, l_th))

    def create_table_v4(self, height, breadth, top_thickness, leg_thickness, roundtop, leg_type):
        h, b, t_th, l_th  = height, breadth, height * top_thickness, breadth * leg_thickness
        rt, l_tp = roundtop, leg_type
        # create table-top
        top_loc, top_scl = (0.0, 0.0, (h - t_th) / 2.0), (1.0, b, t_th)
        cylinder(loc=top_loc, scl=top_scl) if rt else cube(loc=top_loc, scl=top_scl)

        leg_scl = (l_th, l_th, h)
        if l_tp == 'round':
            # create rounded leg
            cylinder(loc=(0.0, 0.0, 0.0), scl=leg_scl)
            cylinder(loc=(0.0, 0.0, -(19.0 * h) / 40.0), scl=(0.5, 0.5, h / 20.0))
        else:
            # create four legs
            a = 1.41 if rt else 1.0
            x, y = (1.0 - a * l_th) / (2 * a), (b - a * l_th) / (2 * a)
            leg_locs = [(x, y, 0.0), (x, -y, 0.0), (-x, y, 0.0), (-x, -y, 0.0)]
            for leg_loc in leg_locs:
                cube(loc=leg_loc, scl=leg_scl)
            # create support between legs
            if l_tp == 'support':
                leg_sprt_scl = (l_th, b / a, l_th) if rt else (l_th, b, l_th)
                cube(loc=(x, 0.0, -h / 3.0), scl=leg_sprt_scl)
                cube(loc=(-x, 0.0, -h / 3.0), scl=leg_sprt_scl)
                cube(loc=(0.0, 0.0, -h / 3.0), scl=((1.0 - a * l_th) / a, l_th, l_th))

    def create_table_v5(self, height, breadth, top_thickness, leg_thickness, roundtop, leg_type):
        h, b, t_th, l_th  = height, breadth, height * top_thickness, breadth * leg_thickness
        rt, l_tp = roundtop, leg_type
        # create table-top
        top_loc, top_scl = (0.0, 0.0, (h - t_th) / 2.0), (1.0, b, t_th)
        cylinder(loc=top_loc, scl=top_scl) if rt else cube(loc=top_loc, scl=top_scl)

        leg_scl = (l_th, l_th, h)
        if l_tp == 'round':
            # create rounded leg
            cylinder(loc=(0.0, 0.0, 0.0), scl=leg_scl)
            cylinder(loc=(0.0, 0.0, -(19.0*h)/40.0), scl=(0.5, 0.5, h/20.0))
        elif l_tp == 'split':
            for x in [0.5 - l_th / 2.0, l_th / 2.0 - 0.5]:
                cube(loc=(x, 0.0, 0.0), scl=(l_th, l_th, h))
                cube(loc=(x, 0.0, (l_th - h) / 2.0), scl=(l_th, b, l_th))
        else:
            # create four legs
            a = 1.41 if rt else 1.0
            x, y = (1.0 - a * l_th) / (2 * a), (b - a * l_th) / (2 * a)
            leg_locs = [(x, y, 0.0), (x, -y, 0.0), (-x, y, 0.0), (-x, -y, 0.0)]
            for leg_loc in leg_locs:
                cube(loc=leg_loc, scl=leg_scl)
            # create support between legs
            if l_tp == 'support':
                leg_sprt_scl = (l_th, b / a, l_th) if rt else (l_th, b, l_th)
                cube(loc=(x, 0.0, -h / 3.0), scl=leg_sprt_scl)
                cube(loc=(-x, 0.0, -h / 3.0), scl=leg_sprt_scl)
                cube(loc=(0.0, 0.0, -h / 3.0), scl=((1.0 - a * l_th) / a, l_th, l_th))

    def get_meshes(self, version, param_vectors):
        meshes = []
        for vector in tqdm(param_vectors, file=sys.stdout, desc='Generating procedural shapes'):
            for obj_name in bpy.data.objects.keys():
                if 'Cube' in obj_name or 'Cylinder' in obj_name:
                    bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
            if version == 1:
                self.create_table_v1(vector[0], vector[1], vector[2], vector[3])
            elif version == 2:
                self.create_table_v2(vector[0], vector[1], vector[2], vector[3], vector[4])
            elif version == 3:
                self.create_table_v3(vector[0], vector[1], vector[2], vector[3], vector[4], vector[5])
            elif version == 4:
                self.create_table_v4(vector[0], vector[1], vector[2], vector[3], vector[4], vector[5])
            elif version == 5:
                self.create_table_v5(vector[0], vector[1], vector[2], vector[3], vector[4], vector[5])
            bpy.ops.object.select_pattern(pattern='Cube*')
            bpy.ops.object.select_pattern(pattern='Cylinder*')
            bpy.ops.object.join()
            verts = np.array([x.co for x in bpy.context.selected_objects[0].to_mesh().vertices])
            faces = np.array([x.vertices for x in bpy.context.selected_objects[0].to_mesh().loop_triangles])
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-3.14 / 2, [1, 0, 0], [0, 0, 0]))
            t = np.sum(mesh.bounding_box.bounds, axis=0) / 2
            mesh.apply_translation(-t)
            s = np.max(mesh.extents)
            mesh.apply_scale(1 / s)
            meshes.append(mesh)
        return meshes


def unit_test(args):
    version = args.version
    num_samples = args.num_samples
    param_def = ParamVectorDef(version)
    proc = TableProcedure()
    '''
    Parameter vectors can be manually defined, such as
    vectors = [
        [0.6, 0.4, 0.05, 0.05, False, 'basic'],
        [0.6, 0.4, 0.05, 0.05, True, 'basic'],
        [0.6, 0.4, 0.05, 0.05, True, 'support'],
        [0.6, 0.4, 0.05, 0.05, True, 'round'],
        [0.6, 0.4, 0.05, 0.05, False, 'split']
    ]
    or we can randomly sample them.
    '''
    vectors = param_def.get_random_vectors(num_samples)
    meshes = proc.get_meshes(version, vectors)
    for mesh in meshes:
        mesh.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=5, help="PML version (1-5)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of examples")
    parsed_args = parser.parse_args()
    unit_test(parsed_args)
