import numpy as np
import pandas as pd

from itertools import chain
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import tensorflow as tf
from tensorflow import keras


def get_train(url, chunk_size):
    data_frame = pd.read_csv(url)
    data_set = data_frame.iloc[:, 1:].values
    data_set = data_set.astype('float64')

    sc_train = MinMaxScaler(feature_range=(0, 1))
    train_data_set = sc_train.fit_transform(data_set)

    reframed_test_data_set = np.array(train_data_set)

    train_x, train_y = reframed_test_data_set[:, :-1], reframed_test_data_set[:, -1]
    train_x = train_x.reshape((train_x.shape[0], chunk_size,14))

    return sc_train,train_x, train_y

def get_test(url, chunk_size):
    data_frame = pd.read_csv(url)
    data_set = data_frame.iloc[:, 1:].values
    data_set = data_set.astype('float64')
    #print(data_set)

    sc = MinMaxScaler(feature_range=(0, 1))
    test_data_set = sc.fit_transform(data_set)

    reframed_test_data_set = np.array(test_data_set)


    test_x, test_y = reframed_test_data_set[:, :-1], reframed_test_data_set[:, -1]
    test_x = test_x.reshape((test_x.shape[0], chunk_size,14))

    return test_x, test_y, reframed_test_data_set,sc

def get_source_data_set(url, chunk_size):
    data_frame = pd.read_csv(url)
    data_set = data_frame.iloc[:, 1:].values
    data_set = data_set.astype('float64')

    sc = MinMaxScaler(feature_range=(0, 1))
    train_data_set = sc.fit_transform(data_set)
    #origin_norm_data = sc.inverse_transform(train_data_set)

    source_data_set = np.array(train_data_set)
    # source_data_set = np.array(series_to_supervised(train_data_set, chunk_size_x, 1).values)
    return source_data_set


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


def plot_img(source_data_set, valid_predict):
    plt.figure(figsize=(24, 8))
    plt.plot(source_data_set[:, -1], c='b')
    plt.plot([x for x in valid_predict], c='g')
    #plt.savefig('plots/test/ATL03_Depth_Ha_Pri_VNIR_ALL_SHIP_Bi_128_0604_SHIP.png', format='png')
    #plt.legend()
    #plt.show()



# define your own chunk size
def main():
    chunk_size = 1
    url_test = r'data/XXX.csv'
    url_train=r'data/XXX.csv'
    sc_train, train_x, train_y=get_train(url=url_train,chunk_size=chunk_size)
    test_x, test_y, reframed_test_data_set,sc=get_test(url=url_test, chunk_size=chunk_size)
    pthfile=r'checkpoints/XXX'
    option = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model=tf.keras.models.load_model(pthfile,options=option)
    model.summary()
    print(model)
    #model = torch.load(pthfile)
    #model.eval()
    print('test_x : ',test_x)
    y_pred = model.predict(test_x)
    #y_Pred = y_pred[np.newaxis, :]
    test_y_1 = test_y[np.newaxis, :]
    y_test = list(chain(*test_y_1))
    #print(test_y)
    y_predict = list(chain(*y_pred))
    #print(y_predict)
    #source_data_set = get_source_data_set(url=url_test, chunk_size=chunk_size)
    #plot_img(source_data_set, y_predict)

    correlation = get_correlation(y_test, y_predict)
    spearman_correlation = get_spearman_correlation(y_test, y_predict)
    mse_pre_src = get_mse_pre_src(y_test, y_predict)
    Rmse = rmse(y_test, y_predict)
    R2 = r2(y_test,y_predict)
    print(mse_pre_src)
    print(correlation)
    print(spearman_correlation)
    print('The value of R2 is', R2)
    print('The value of Rmse is', Rmse)
    
    max_standard_train = sc_train.data_max_[-1]
    min_standard_train = sc_train.data_min_[-1]

    max_standard_test=sc.data_max_[-1]
    min_standard_test=sc.data_min_[-1]


    print('max_standard_train : ',max_standard_train)
    print('min_standard_train : ',min_standard_train)


    real_predict = y_pred * (max_standard_train - min_standard_train) + min_standard_train
    real_y = test_y * (max_standard_test - min_standard_test) + min_standard_test

    filename = 'results/XXX.txt'
    #np.savetxt(filename, np.column_stack([y_predict,real_predict]))
    np.savetxt(filename, np.column_stack([real_y,real_predict]))




if __name__ == '__main__':
    main()
