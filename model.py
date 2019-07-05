from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


class Model:
    #the model
    model= DummyClassifier(strategy='stratified', random_state=None, constant=None)
    # classification=0;regression=1
    task=0

    def __init__(self, task, modelName):
        self.task = task
        #create the model
        if task ==0:
            if modelName == "NB":
                self.model = GaussianNB()
            elif modelName == "KNN":
                self.model = KNeighborsClassifier()
            elif modelName == "SVM":
                self.model = SVC()
            elif modelName == "C45":
                self.model = tree.DecisionTreeClassifier
            elif modelName == "MultiLabelNB":
                self.model =  BinaryRelevance(GaussianNB())
            else:
                print("YOU CHOSE WRONG MODEL FOR CLASSIFICATION!")
        else:
            if modelName == "LR":
                self.model = linear_model.LinearRegression()
            elif modelName == "M5":
                self.model = tree.DecisionTreeRegressor
            elif modelName == "KNN":
                self.model = KNeighborsRegressor()
            else:
                print("YOU CHOSE WRONG MODEL FOR REGRESSION!")
                self.model = linear_model.LinearRegression()

    def train(self, data, multilabel=False):
        print("training...")
        scoring = 'accuracy'
        if self.task==1:
            scoring="neg_mean_squared_error"
        if multilabel:
            labels = list(map(lambda x: x.split(','), data.values[:, -1]))
            classes = set()
            for label in labels:
                classes |= set(label)
            print(classes)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            labels_onehot = np.array([np.sum([classes_dict.get(l) for l in label], axis=0) for label in labels],
                                     dtype=np.int32)
            labels = labels_onehot
        else:
            labels = list(map(lambda x: x.split(','), data.values[:, -1]))
            classes = set()
            for label in labels:
                classes |= set(label)
            print(classes)
            labels = data.iloc[:, -1]
        scores = cross_val_score(self.model, data.iloc[:, 1:-1], labels, cv=10, scoring=scoring)
        print(scoring, np.mean(scores))
