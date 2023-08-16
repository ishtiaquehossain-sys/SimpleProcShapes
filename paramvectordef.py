import random
import numpy as np
from itertools import product


class Parameter:
    def __init__(self, paramtype, data):
        self.paramtype = paramtype
        self.data = data


class ParamVectorDef:
    def __init__(self, version):
        self.params = []

        self.params.append(Parameter('scalar', [(0.5, 1.0, 8), (0.3, 1.0, 8), (0.05, 0.25, 5), (0.05, 0.25, 5)]))
        if version > 1:
            self.params.append(Parameter('binary', None))
        if version > 2:
            self.params.append(Parameter('type', ['basic', 'support']))
            if version > 3:
                self.params[-1].data.append('round')
            if version > 4:
                self.params[-1].data.append('split')

    def get_random_vectors(self, max_len=3000):
        param_domain = []
        for param in self.params:
            if param.paramtype == 'scalar':
                for elem in param.data:
                    param_domain.append(np.linspace(elem[0], elem[1], elem[2]))
            elif param.paramtype == 'binary':
                param_domain.append([False, True])
            elif param.paramtype == 'type':
                param_domain.append(param.data)
        if not param_domain:
            param_vectors = []
        else:
            param_vectors = [p for p in product(*param_domain)]
        random.shuffle(param_vectors)
        if max_len is not None:
            if len(param_vectors) > max_len:
                param_vectors = random.choices(param_vectors, k=max_len)
        return param_vectors

    def encode(self, vectors):
        encoded_vectors = []
        index = 0
        for param in self.params:
            paramtype = param.paramtype
            if paramtype == 'scalar':
                col_length = len(param.data)
            else:
                col_length = 1
            if paramtype == 'scalar' or paramtype == 'binary':
                data = np.array([x[index:index+col_length] for x in vectors], dtype=float)
                if paramtype == 'scalar':
                    min_vals = np.array([x[0] for x in param.data], ndmin=2)
                    max_vals = np.array([x[1] for x in param.data], ndmin=2)
                    data = (data - min_vals) / (max_vals - min_vals)
                encoded_vectors.append(data)
            elif paramtype == 'type':
                ohe_coded = np.array(
                    [[0 if x[index] != y else 1 for y in param.data] for x in vectors], dtype=float
                )
                encoded_vectors.append(ohe_coded)
            index = index + col_length
        return encoded_vectors

    def decode(self, vectors):
        collector = []
        for i, param in enumerate(self.params):
            paramtype = param.paramtype
            data = vectors[i]
            if paramtype == 'scalar':
                min_vals = np.array([x[0] for x in param.data], ndmin=2)
                max_vals = np.array([x[1] for x in param.data], ndmin=2)
                data = data * (max_vals - min_vals) + min_vals
            elif paramtype == 'binary':
                data = np.isclose(data, 1.0)
            elif param.paramtype == 'type':
                data = np.expand_dims(np.array(param.data)[np.argmax(data, axis=1)], axis=1)
            collector.append(data.tolist())
        decoded_vectors = []
        for i in range(len(collector[0])):
            vector = []
            for j in range(len(self.params)):
                vector.extend(collector[j][i])
            decoded_vectors.append(vector)
        return decoded_vectors


def unit_test(version, num_samples):
    param_def = ParamVectorDef(version)
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
    enc_vectors = param_def.encode(vectors)
    dec_vectors = param_def.decode(enc_vectors)
    print(vectors)
    print(dec_vectors)


if __name__ == '__main__':
    unit_test(version=5, num_samples=5)
