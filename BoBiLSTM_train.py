import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Bidirectional,Conv1D
from keras.layers import Dropout
from pandas import DataFrame
from pandas import concat
from itertools import chain
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import tensorflow as tf
from tensorflow import keras
from sklearn.externals import joblib



def get_train_valid(url, chunk_size):
    data_frame = pd.read_csv(url)
    data_set = data_frame.iloc[:, 1:].values
    data_set = data_set.astype('float64')
    #print(data_set)

    sc = MinMaxScaler(feature_range=(0, 1))
    train_data_set = sc.fit_transform(data_set)


    reframed_train_data_set = np.array(train_data_set)


    train_days = int(len(train_data_set) * 0.8)
    valid_days = int(len(train_data_set) * 0.2)

    train = reframed_train_data_set[:train_days, :]
    valid = reframed_train_data_set[train_days:train_days + valid_days, :]

    train_x, train_y = train[:, :-1], train[:, -1]
    valid_x, valid_y = valid[:, :-1], valid[:, -1]
    #print(valid_x)
    #print(valid_y)
    #print(train_x)

    train_x = train_x.reshape((train_x.shape[0], chunk_size, 14))
    valid_x = valid_x.reshape((valid_x.shape[0], chunk_size, 14))


    return train_x, train_y, valid_x, valid_y,reframed_train_data_set,sc

def Normalize(data):
    sc = MinMaxScaler(feature_range=(0, 1))
    train_data_set = sc.fit_transform(sc)
    return train_data_set

def get_source_data_set(url, chunk_size):
    data_frame = pd.read_csv(url)
    data_set = data_frame.iloc[:, 1:].values
    data_set = data_set.astype('float64')

    sc = MinMaxScaler(feature_range=(0, 1))
    train_data_set = sc.fit_transform(data_set)

    source_data_set = np.array(train_data_set)
    # source_data_set = np.array(series_to_supervised(train_data_set, chunk_size_x, 1).values)
    return source_data_set

def lstm_model(url, train_x, label_y, valid_x, valid_y, input_epochs, input_batch_size,
               chunk_size):
    model = Sequential()
    
    #model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(train_x.shape[1], train_x.shape[2])))
    #model.add(Dropout(0.5))

    #model.add(Conv1D(filters=160, kernel_size=1, activation='ReLU',input_shape=(train_x.shape[1], train_x.shape[2])))  # , padding = 'same'
    #model.add(Dropout(0.5))

    
    #model.add(LSTM(128, return_sequences=False))
    #model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, activation='tanh', input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=False)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')


    res = model.fit(train_x, label_y, epochs=input_epochs, batch_size=input_batch_size,
                    validation_data=(valid_x, valid_y), verbose=2, shuffle=False)

    # prediction Generates output predictions for the input samples.
    train_predict = model.predict(train_x)
    valid_predict = model.predict(valid_x)
    valid_y_1 = valid_y[np.newaxis, :]

    valid = list(chain(*valid_y_1))
    valid_predict_1 = list(chain(*valid_predict))

    save_filename = 'checkpoints/XXX'
    save_options=tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    model.save(save_filename,options=save_options)
    print('Saved as %s' % save_filename)

    # source_data_set = get_reframed_train_data_set(url=url, chunk_size_x=chunk_size_x)
    source_data_set = get_source_data_set(url=url, chunk_size=chunk_size)

    plt.plot(res.history['loss'], label='train')
    plt.show()
    print(model.summary())
    plot_img(source_data_set,train_predict, valid_predict_1)

    correlation = get_correlation(valid, valid_predict_1)
    spearman_correlation = get_spearman_correlation(valid, valid_predict_1)
    mse_pre_src = get_mse_pre_src(valid, valid_predict_1)
    Rmse=rmse(valid,valid_predict_1)
    R2=r2(valid,valid_predict_1)
    return mse_pre_src, correlation, spearman_correlation,Rmse,R2,model,valid,valid_predict_1,valid_predict,valid_y

from sklearn.metrics import r2_score,mean_squared_error

def r2(y1,y2):
    r2 = r2_score(y1, y2)
    return r2

def rmse(y1, y2):
    RMSE=np.sqrt(mean_squared_error(y1, y2))
    return RMSE

def get_mse_pre_src(valid_data,predict_data):
    mse = mean_squared_error(valid_data, predict_data)
    return mse


def get_correlation(valid_data, predict_data):
    ans = np.corrcoef(np.array(valid_data), np.array(predict_data))
    return ans


def get_spearman_correlation(valid_data, predict_data):
    df2 = pd.DataFrame({'real': valid_data, 'prediction': predict_data})
    return df2.corr('spearman')


def plot_img(source_data_set,train_predict, valid_predict):
    plt.figure(figsize=(24, 8))
    plt.plot(source_data_set[:, -1], c='b')
    plt.plot([x for x in train_predict], c='g')
    plt.plot([None for _ in train_predict] + [x for x in valid_predict], c='y')
    #plt.plot([None for _ in train_predict] + [None for _ in valid_predict] + [x for x in test_predict], c='r')
    plt.savefig('plots/XXX.png', format='png')
    plt.legend()
    plt.show()


# define your own chunk size and epoch times
def main():
    chunk_size = 1
    input_epochs = 1000
    input_batch_size = 64
    url = r'data/XXX.csv'
    train_x, label_y, valid_x, valid_y, reframed,sc = \
        get_train_valid(url=url, chunk_size=chunk_size)
    mse_pre_src, correlation, spearman_correlation,Rmse,R2,model,valid,valid_predict_1,valid_predict,valid_y = lstm_model(url, train_x, label_y,
                                                                valid_x, valid_y,
                                                                input_epochs, input_batch_size,
                                                                chunk_size=chunk_size)
    print(mse_pre_src)
    print(correlation)
    print(spearman_correlation)
    print('The value of R2 is',R2)
    print('The value of Rmse is',Rmse)
    
    max_standard = sc.data_max_[-1]
    min_standard = sc.data_min_[-1]

    real_predict = valid_predict * (max_standard - min_standard) + min_standard
    real_y = valid_y * (max_standard - min_standard) + min_standard
    filename = 'results/XXX.txt'
    np.savetxt(filename, np.column_stack([real_y,real_predict]))
    # print(reframed)


if __name__ == '__main__':
    main()
