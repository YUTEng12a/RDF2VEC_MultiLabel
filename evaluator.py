import pandas as pd
import numpy as np
import data_manager
import model


print("--------------------processing cora data: NB--------------------")

dataset = 'cora'
vectors_file = './data/{}/{}.content'.format(dataset, dataset)

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "NB"

# data manager
data = data_manager.data_manager.readData(vectors_file, dataset)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model = model.Model(task, modelName)
# train and print score
model.train(data)


print("--------------------processing cora data: SVM--------------------")

dataset = 'cora'
vectors_file = './data/{}/{}.content'.format(dataset, dataset)

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "SVM"

# data manager
data = data_manager.data_manager.readData(vectors_file, dataset)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model = model.Model(task, modelName)
# train and print score
model.train(data)

print("--------------------processing FB15K237 data: mult NB--------------------")

dataset = 'FB15K237'
vectors_file = './data/{}/{}.content'.format(dataset, dataset)

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "MultiLabelNB"

# data manager
data = data_manager.data_manager.readData(vectors_file, dataset)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model = model.Model(task, modelName)
# train and print score
model.train(data, True)


print("--------------------processing FB15K237 data: mult SVM--------------------")

dataset = 'FB15K237'
vectors_file = './data/{}/{}.content'.format(dataset, dataset)

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "MultiLabelSVM"

# data manager
data = data_manager.data_manager.readData(vectors_file, dataset)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model = model.Model(task, modelName)
# train and print score
model.train(data, True)

print("--------------------processing WN18RR data: mult NB--------------------")

dataset = 'WN18RR'
vectors_file = './data/{}/{}.content'.format(dataset, dataset)

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "MultiLabelNB"

# data manager
data = data_manager.data_manager.readData(vectors_file, dataset)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model = model.Model(task, modelName)
# train and print score
model.train(data, True)


print("--------------------processing WN18RR data: mult SVM--------------------")

dataset = 'WN18RR'
vectors_file = './data/{}/{}.content'.format(dataset, dataset)

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "MultiLabelSVM"

# data manager
data = data_manager.data_manager.readData(vectors_file, dataset)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model = model.Model(task, modelName)
# train and print score
model.train(data, True)

print("--------------------processing WN18RR_sub30000 data: mult NB--------------------")

dataset = 'WN18RR_sub30000'
vectors_file = './data/{}/{}.content'.format(dataset, dataset)

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "MultiLabelNB"

# data manager
data = data_manager.data_manager.readData(vectors_file, dataset)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model = model.Model(task, modelName)
# train and print score
model.train(data, True)


print("--------------------processing WN18RR_sub30000 data: mult SVM--------------------")

dataset = 'WN18RR_sub30000'
vectors_file = './data/{}/{}.content'.format(dataset, dataset)

# classification=0;regression=1
# in case of regression, neg_mean_squared_error is used; to calculate RMSE simply calculate the root
task = 0
# model option: NB, SVM, KNN, LR, M5
modelName = "MultiLabelSVM"

# data manager
data = data_manager.data_manager.readData(vectors_file, dataset)
data = data.sample(frac=1).reset_index(drop=True)

# initialize the model
model = model.Model(task, modelName)
# train and print score
model.train(data, True)
                        
