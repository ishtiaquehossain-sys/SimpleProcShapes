import sys
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
import trimesh

CURRENT_DIR = Path(__file__).parent


class MeshEncoder:
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        self.temp_dir_path = Path(tempfile.mkdtemp())
        self.file_name = uuid.uuid4()
        self.temp_path = self.temp_dir_path / "{}.obj".format(self.file_name)

        self.mesh.export(self.temp_path.as_posix())

    def get_path(self) -> str:
        return self.temp_path.with_suffix("").as_posix()

    def align_mesh(self):
        process = subprocess.Popen(
            ["./3DAlignment", self.temp_path.with_suffix("").as_posix()],
            cwd=(CURRENT_DIR / "Executable").as_posix(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output, err = process.communicate()
        if len(err) > 0:
            print(err)
            sys.exit(1)

        features = np.array([int(x) for x in output.decode('utf-8').split(' ')[:-1]])
        return features

    def __del__(self):
        shutil.rmtree(self.temp_dir_path.as_posix())


class LightFieldDistance:
    def get_descriptor(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        features = MeshEncoder(vertices, triangles).align_mesh()
        return features

    def get_distance(self, features_1: np.ndarray, features_2: np.ndarray) -> float:
        with open(os.path.join(CURRENT_DIR, 'Executable', 'coeffs'), 'wb') as f:
            arr = bytearray(features_1.astype(np.uint8))
            f.write(arr)
            arr = bytearray(features_2.astype(np.uint8))
            f.write(arr)
            f.close()
        process = subprocess.Popen(
            ["./Distance"],
            cwd=(CURRENT_DIR / "Executable").as_posix(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, err = process.communicate()
        distance = int(output.decode('utf-8'))
        os.remove(CURRENT_DIR / 'Executable' / 'coeffs')

        return distance
