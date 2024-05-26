import pandas as pd
import numpy as np
import tensorflow as tf

tf.config.list_physical_devices('GPU')


def Classify_feature_variable(data):
    """
    data: Data read from the folder
    return: Temporal explanatory variables and spatial explanatory variables

    """
    features_times = []  # list of feature variables that vary with time
    features_non_times = []  # list of feature variables that do not vary with time
    name_pre = 1
    for feat in data.columns:
        if feat[-4:].isdigit():
            name = feat[:-4]
            year = int(feat[-4:])
            if name_pre != name:
                features_times.append(name)
                name_pre = name
        else:
            name = feat
            year = -1
            features_non_times.append(name)
    return features_times, features_non_times


def construct_combined_data(data, features_times, features_non_times, id=100,start_year=1988, end_year=2012, k=0,group=50):
    """
    data: Data read from the folder
    features_times: Temporal explanatory variables
    features_non_times: Spatial explanatory variables
    id: Number of samples
    start_year: Start year
    end_year: End year
    return: A tensor composed of (end_year - start_year) tensors, each with a shape of (number of samples, number of features)

    """
    combined_data = []
    for year in range(start_year, end_year):
        time_varying_data = np.zeros((id, len(features_times)))
        non_time_varying_data = np.zeros((id, len(features_non_times)))
        for i in range(id):
            for j, feat in enumerate(features_times):
                feat_col = feat + str(year)
                time_varying_data[i, j] = data.loc[i + k * group, feat_col]
            for j, feat in enumerate(features_non_times):
                feat_col = feat
                non_time_varying_data[i, j] = data.loc[i + k * group, feat_col]
        combined_data_step = np.concatenate(
            (time_varying_data, non_time_varying_data), axis=1)
        combined_data.append(combined_data_step)
    return combined_data

def constant_target(process_data, col=9):
    """
    process_data: List of 2D tensors to be processed
    col: Index of the column containing the dependent variable
    return: List of dependent variable tensors

    """
    target = []  # 因变量
    for tensor in process_data:
        target_per_year = tensor[:, col:(col + 1)]
        target.append(target_per_year)
    return target


def test_variable(process_data, col=9):
    """
    process_data: List of 2D tensors to be processed
    col: Index of the column containing the dependent variable
    return: List of explanatory variable tensors

    """
    features = []  # 因变量
    for tensor in process_data:
        features_per_year = np.delete(tensor, col, axis=1)
        features.append(features_per_year)
    return features


def expand_tensor_dimension(data):
    """
    data: List of 2D tensors to be processed
    return: List of 3D tensors, with each 3D tensor having the shape (rows of 2D list, columns of 2D list, 1)

    """
    tensor_3d = []
    for tensor in data:
        new_tensor = np.expand_dims(tensor, axis=-1)
        tensor_3d.append(new_tensor)
    return tensor_3d


def trans_to_asarray(input_feature, input_target):
    """
    featur: Explanatory variables to be transformed
    target: Dependent variable to be transformed
    return: Transformed explanatory variables and dependent variable

    """
    model_feature = np.asarray(input_feature)
    model_target = np.asarray(input_target)

    return model_feature, model_target



#load data

path = r'../Data/TAS_AfterDataPredeal.xlsx'
data = pd.read_excel(path)
for j in range(1988, 2009):
    train_features = np.empty((295, 2, 100, 49, 1))
    train_target = np.empty((295, 2, 100, 1, 1))
    for i in range(0, 295):
        group = 100
        features_times, features_non_times = Classify_feature_variable(data)
        train_3d_features = construct_combined_data(data, features_times,features_non_times,
                                                    start_year=j, end_year=j+2,id=group, k=i,group=group)
        train_3d_target = construct_combined_data(data, features_times,features_non_times,
                                                  start_year=j+1, end_year=j+3,id=group, k=i, group=group)
        train_3d_target = constant_target(train_3d_target, col=9)
        train_3d_features = test_variable(train_3d_features, col=9)
        train_4d_features = expand_tensor_dimension(train_3d_features)
        train_4d_target = expand_tensor_dimension(train_3d_target)
        train_features[i] = train_4d_features
        train_target[i] = train_4d_target
    train_model_features = np.stack(train_features, axis=0)
    train_model_target = np.stack(train_target, axis=0)
    np.save(f'train_model_features_{j}.npy', train_model_features)
    np.save(f'train_model_target_{j}.npy', train_model_target)

for j in range(2009, 2013):
    val_features = np.empty((295, 2, 100, 49, 1))
    val_target = np.empty((295, 2, 100, 1, 1))
    for i in range(0, 295):
        group = 100
        features_times, features_non_times = Classify_feature_variable(data)
        val_3d_features = construct_combined_data(data, features_times,features_non_times,
                                                  start_year=j,end_year=j+2,id=group,k=i,
                                                  group=group)
        val_3d_target = construct_combined_data(data,features_times,features_non_times,
                                                start_year=j+1,end_year=j+3,id=group,k=i,
                                                group=group)
        val_3d_target = constant_target(val_3d_target,col=9)
        val_3d_features = test_variable(val_3d_features,col=9)
        val_4d_features = expand_tensor_dimension(val_3d_features)
        val_4d_target = expand_tensor_dimension(val_3d_target)
        val_features[i] = val_4d_features
        val_target[i] = val_4d_target
    val_model_features = np.stack(val_features,axis=0)
    val_model_target = np.stack(val_target,axis=0)
    np.save(f'val_model_features_{j}.npy',val_model_features)
    np.save(f'val_model_target_{j}.npy',val_model_target)

print(train_model_features.shape)
print(train_model_target.shape)
print(val_model_features.shape)
print(val_model_target.shape)
print('File saved')

