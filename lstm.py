import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import pandas_datareader.data as web
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint


stock_name = '^IXIC'
seq_len = 22
d = 0.3
shape = [4, seq_len, 1]  # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 200


def plot_stock(stock_name):
    df = get_stock_data(stock_name, normalize=True)
    print(df.head())
    plt.plot(df['Adj Close'], color='red', label='Adj Close')
    plt.legend(loc='best')
    plt.show()


def get_stock_data(stock_name, normalize=True):
    start = datetime.datetime(1980, 1, 2)
    end = datetime.date.today()
    df = web.DataReader(stock_name, "yahoo", start, end)
    df.drop(['Volume', 'Close'], 1, inplace=True)

    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
        df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
    return df


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    # data = stock.as_matrix()
    data = stock.iloc[:, :].values
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days

    result = np.array(result)
    row = round(0.9 * result.shape[0])  # 90% split

    train = result[:int(row), :]  # 90% date
    X_train = train[:, :-1]  # all data until day m
    y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]


def build_model2(layers, neurons, d):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[2]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
    model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
    # model = load_model('LSTM_Stock_prediction-20200426.h5')
    adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def show_loss():
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


def percentage_difference(model, X_test, y_test):
    percentage_diff = []

    p = model.predict(X_test)
    for u in range(len(y_test)):  # for each data index in test data
        pr = p[u][0]  # pr = prediction on day u

        percentage_diff.append((pr - y_test[u] / pr) * 100)
    # print("percentage_difference =", np.mean(percentage_diff))
    return p


def denormalize(stock_name, normalized_value):
    start = datetime.datetime(2000, 1, 1)
    end = datetime.date.today()
    df = web.DataReader(stock_name, "yahoo", start, end)

    df = df['Adj Close'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)

    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


def plot_result(stock_name, normalized_value_p, normalized_value_y_test):
    newp = denormalize(stock_name, normalized_value_p)
    newy_test = denormalize(stock_name, normalized_value_y_test)
    plt.plot(newp, color='red', label='Prediction')
    plt.plot(newy_test, color='blue', label='Actual')
    plt.legend(loc='best')
    plt.title('The test result for {}'.format(stock_name))
    plt.xlabel('Days')
    plt.ylabel('Adjusted Close')
    plt.show()


def quick_measure(stock_name, seq_len, d, shape, neurons, epochs):
    df = get_stock_data(stock_name)
    X_train, y_train, X_test, y_test = load_data(df, seq_len)
    model = build_model2(shape, neurons, d)
    model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1, verbose=1)
    # model.save('LSTM_Stock_prediction-20170429.h5')
    trainScore, testScore = model_score(model, X_train, y_train, X_test, y_test)
    return trainScore, testScore


# df = get_stock_data(stock_name, normalize=True)
# X_train, y_train, X_test, y_test = load_data(df, seq_len)
# plot_stock(stock_name)     # 生成df示例图与股价走势图
# model = build_model2(shape, neurons, d)
# history = model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1, verbose=1)
# model.save('LSTM_Stock_prediction-20200426.h5')
#
# show_loss()
# model_score(model, X_train, y_train, X_test, y_test)
# p = percentage_difference(model, X_test, y_test)
# plot_result(stock_name, p, y_test)


#########################################################################
# 接下来优化dropout
dlist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
neurons_LSTM = [32, 64, 128, 256, 512, 1024, 2048]
dropout_result = {}
for d in dlist:
    trainScore, testScore = quick_measure(stock_name, seq_len, d, shape, neurons, epochs)
    dropout_result[d] = testScore

min_val = min(dropout_result.values())
min_val_key = [k for k, v in dropout_result.items() if v == min_val]
print(dropout_result)
print(min_val_key)

#################################
# 可视化dropout

lists = sorted(dropout_result.items())
x, y = zip(*lists)
plt.plot(x, y)
plt.title('Finding the best hyperparameter')
plt.xlabel('Dropout')
plt.ylabel('Mean Square Error')
plt.show()

##########################################################################
# 优化神经元个数
neuronlist1 = [32, 64, 128, 256, 512]
neuronlist2 = [16, 32, 64]
neurons_result = {}

for neuron_lstm in neuronlist1:
    neurons = [neuron_lstm, neuron_lstm]
    for activation in neuronlist2:
        neurons.append(activation)
        neurons.append(1)
        trainScore, testScore = quick_measure(stock_name, seq_len, d, shape, neurons, epochs)
        neurons_result[str(neurons)] = testScore
        neurons = neurons[:2]
######################################
# 可视化
lists = sorted(neurons_result.items())
x, y = zip(*lists)

plt.title('Finding the best hyperparameter')
plt.xlabel('neurons')
plt.ylabel('Mean Square Error')

plt.bar(range(len(lists)), y, align='center')
plt.xticks(range(len(lists)), x)
plt.xticks(rotation=90)

plt.show()


###########################################################################


