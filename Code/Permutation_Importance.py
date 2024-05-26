import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Reshape, Input, BatchNormalization, Conv2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Attention_Module import dual_attentiont1, dual_attentiont2, dual_attentiont3, dual_attentiont4
from ResConvLSTM import ResConvLSTM2D
import time


tf.config.list_physical_devices('GPU')
tf.config.run_functions_eagerly(True)

def create_train_ResConvLSTM_model(num=100):
    inputs = Input(shape=(2, num, 49, 1))
    x1 = ResConvLSTM2D(filters=128, kernel_size=(3, 3), return_sequences=True,
                      padding='same', input_shape=(2, 100, 49, 1))(inputs)
    x2 = ResConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=True,
                      padding='same')(x1)
    x2out = []
    for t in range(2):
        xtoRes = x2[:,t,:,:,:]
        if t==0:
            xt = dual_attentiont1(xtoRes,64)
            xt =  tf.expand_dims(xt, axis=1)
        else:
            xt = dual_attentiont2(xtoRes,64)
            xt =  tf.expand_dims(xt, axis=1)
        x2out.append(xt)
    x2next = tf.concat(x2out,axis=1)
    x1 = Conv2D(64, (1, 1), activation='relu')(x1)
    x3 = x2next + x1
    x3 = BatchNormalization()(x3)
    x3 = Activation(activation='relu')(x3)
    x4 = ResConvLSTM2D(filters=32, kernel_size=(3, 3), return_sequences=True,
                      padding='same')(x3)
    x4out = []
    for t in range(2):
        x_conv_to_Res = x4[:,t,:,:,:]
        if t==0:
            xconvt = dual_attentiont3(x_conv_to_Res,32)
            xconvt =  tf.expand_dims(xconvt, axis=1)
        else:
            xconvt = dual_attentiont4(x_conv_to_Res,32)
            xconvt =  tf.expand_dims(xconvt, axis=1)
        x4out.append(xconvt)
    x4next = tf.concat(x4out,axis=1)
    x3 = Conv2D(32, (1, 1), activation='relu')(x3)
    x5 = x4next + x3
    x5 = BatchNormalization()(x5)
    x5 = Activation(activation='relu')(x5)
    x = ResConvLSTM2D(filters=1, kernel_size=(3, 3), return_sequences=False,
                      padding='same')(x5)

    x = Flatten()(x)
    x = Dense(units=1 * num * 1 * 1, activation='relu')(x)
    outputs = Reshape((1, num, 1, 1))(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

def Calcu_rmse_R2_11_14(cal_flat_true_y, cal_flat_pre_y):
    """
    pre_y: Predicted values
    val_y: Actual values
    return: Computed RMSE and pseudo R-squared

    """
    MSE = mean_squared_error(cal_flat_true_y, cal_flat_pre_y)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(cal_flat_true_y, cal_flat_pre_y)
    return MSE, RMSE, R2

def Flattern(pre_y, true_y):
    """
    pre_y: Predicted values
    val_y: Actual values
    return: Computed RMSE and pseudo R-squared

    """
    cal_true_y = true_y[:, -1:, :, :, :]
    cal_pre_y = pre_y[:, -1:, :, :, :]
    cal_flat_true_y = cal_true_y.reshape(-1)
    cal_flat_pre_y = cal_pre_y.reshape(-1)
    MSE, RMSE, R2 = Calcu_rmse_R2_11_14(cal_flat_pre_y, cal_flat_true_y)
    return MSE, RMSE, R2


cal_feature_model = create_train_ECAConvLSTM_model()
cal_feature_model.summary()
cal_feature_model.load_weights(f'train_model_weights_2008.h5')

for year in range(2009, 2013):
    #Original rmse, r2
    feature_score = []
    print('Testing:{}'.format(year))
    val_model_features = np.load(f'val_model_features_{year}.npy')
    val_model_target = np.load(f'val_model_target_{year}.npy')
    val_model_target = val_model_target[:, -1:, :, :, :]
    pre_y = cal_feature_model.predict(val_model_features, batch_size=4)
    MSE_ori, RMSE_ori, R2_ori = Flattern(pre_y, val_model_target)

    #Importance of each variable
    for i in range(val_model_features.shape[3]):
        print(i)
        new_features = np.copy(val_model_features)
        new_features[:, :, :, i, :] = np.random.permutation(new_features[:, :, :, i, :])
        pre_y = cal_feature_model.predict(new_features, batch_size=4)
        MSE, RMSE, R2 = Flattern(pre_y, val_model_target)
        feature_score.append(MSE_2011 - MSE)

    print(feature_score)
    df = pd.DataFrame(feature_score, columns=['Score'])
    df.to_csv(f'feature_score_MSE_{year}.csv', index=False)
