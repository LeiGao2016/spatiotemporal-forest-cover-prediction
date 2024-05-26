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


start_time = time.time()

train_model = create_train_ResConvLSTM_model()
train_model.summary()

for year in range(1988, 2009):
    print('Training:{}'.format(year))
    train_model_features = np.load(f'train_model_features_{year}.npy')
    train_model_target = np.load(f'train_model_target_{year}.npy')
    train_model_target = train_model_target[:, -1:, :, :, :]
    print(np.shape(train_model_target))
    train_model.fit(train_model_features, train_model_target, epochs=50,
                    batch_size=4)
    if year == 2008:
        train_model.save_weights(f'train_model_weights_{year}.h5')
    del train_model_features
    del train_model_target

test_model = create_train_ResConvLSTM_model()
test_model.summary()
print('Testing')
for year in range(2009, 2013):
    test_model.load_weights(f'train_model_weights_2008.h5')
    val_model_features = np.load(f'val_model_features_{year}.npy')
    val_model_target = np.load(f'val_model_target_{year}.npy')
    val_model_target = val_model_target[:, -1:, :, :, :]
    print('Testing:{}'.format(year))
    pre_y = test_model.predict(val_model_features, batch_size=4)
    np.save(f'pre_y_{year + 2}.npy', pre_y)

true_y_2011 = np.load('val_model_target_2009.npy')
true_y_2012 = np.load('val_model_target_2010.npy')
true_y_2013 = np.load('val_model_target_2011.npy')
true_y_2014 = np.load('val_model_target_2012.npy')
pre_y_2011 = np.load('pre_y_2011.npy')
pre_y_2012 = np.load('pre_y_2012.npy')
pre_y_2013 = np.load('pre_y_2013.npy')
pre_y_2014 = np.load('pre_y_2014.npy')
MSE_2011, RMSE_2011, R2_2011 = Flattern(pre_y_2011, true_y_2011)
MSE_2012, RMSE_2012, R2_2012 = Flattern(pre_y_2012, true_y_2012)
MSE_2013, RMSE_2013, R2_2013 = Flattern(pre_y_2013, true_y_2013)
MSE_2014, RMSE_2014, R2_2014 = Flattern(pre_y_2014, true_y_2014)
aveRMSE = (RMSE_2011 + RMSE_2012 + RMSE_2013 + RMSE_2014) / 4
aveR2 = (R2_2011 + R2_2012 + R2_2013 + R2_2014) / 4
print(
    'MSE_2011={},RMSE_2011={}, R2_2011={}'.format(MSE_2011, RMSE_2011, R2_2011))
print(
    'MSE_2012={},RMSE_2012={}, R2_2012={}'.format(MSE_2012, RMSE_2012, R2_2012))
print(
    'MSE_2013={},RMSE_2013={}, R2_2013={}'.format(MSE_2013, RMSE_2013, R2_2013))
print(
    'MSE_2014={},RMSE_2014={}, R2_2014={}'.format(MSE_2014, RMSE_2014, R2_2014))
print('aveRMSE = {},aveR2 = {}'.format(aveRMSE, aveR2))

end_time = time.time()

total_time = end_time - start_time
print(f"The total running time of the program is {total_time} seconds.")