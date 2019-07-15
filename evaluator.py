import pandas as pd
import numpy as np
import data_manager
import model


print("--------------------processing cora data--------------------")
# parameters

# input file names
# the expected label for classification should be named "label"; for regressions should be called "rating"
# the id should be named "DBpedia_URI15"
vectors_file = './data/cora/cora.content'

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "SVM"

# data manager
data = data_manager.data_manager.readData(vectors_file)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model1 = model.Model(task, modelName)
# train and print score
model1.train(data)

print("--------------------processing FB15K237 data--------------------")
# parameters

# input file names
# the expected label for classification should be named "label"; for regressions should be called "rating"
# the id should be named "DBpedia_URI15"
vectors_file = './data/FB15K237/FB15K237.content'

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "MultiLabelSVM"

# data manager
data = data_manager.data_manager.readData(vectors_file)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model2 = model.Model(task, modelName)
# train and print score
model2.train(data, True)

print("--------------------processing WN18RR data--------------------")
# parameters

# input file names
# the expected label for classification should be named "label"; for regressions should be called "rating"
# the id should be named "DBpedia_URI15"
vectors_file = './data/WN18RR/WN18RR.content'

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "MultiLabelSVM"

# data manager
data = data_manager.data_manager.readData(vectors_file)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model3 = model.Model(task, modelName)
# train and print score
model3.train(data, True)
