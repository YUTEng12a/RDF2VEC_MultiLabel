import pandas as pd
import numpy as np
import os


class data_manager:
    def __init__(self):
        print("initiated")

    @staticmethod
    def readData(vectors_file, dataset):
        if dataset == 'cora':
            vectors = pd.read_csv(vectors_file, "\t", header=None)
        else:
            if os.path.exists("./data/{}/{}.dele".format(dataset, dataset)):
                idx_features_labels = np.genfromtxt("./data/{}/{}.content".format(dataset, dataset), dtype=np.dtype(str))
                names = idx_features_labels[:, 0]
                idx = np.array(idx_features_labels[:, 1], dtype=np.int32)
                idx_map = {j: i for i, j in enumerate(idx)}
                delete_entities_names = np.genfromtxt("./data/{}/{}.dele".format(dataset, dataset), dtype=np.dtype(str))
                delete_entities_arg = np.array([np.where(names == ent_names)[0][0] for ent_names in delete_entities_names])
                delete_entities_idx = list(map(idx_map.get, delete_entities_arg))
                new_idx = []
                for index in range(len(idx_map)):
                    if not index in delete_entities_idx:
                        new_idx.append(index)
                vectors = pd.read_csv(vectors_file, "\t", header=None).loc[new_idx, 1:]
            else:
                vectors = pd.read_csv(vectors_file, "\t", header=None).loc[:, 1:]
        return vectors
