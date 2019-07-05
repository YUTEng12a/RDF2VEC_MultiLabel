import pandas as pd
import numpy as np

class data_manager:
    def __init__(self):
        print("initiated")

    @staticmethod
    def readData(vectors_file):
        vectors = pd.read_csv(vectors_file, "\t", header=None)
        # print(vectors)
        return vectors
