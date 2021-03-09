import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import json


def main():
    # read the stock information
    df = pd.read_csv("IBM.csv")
    print("Number of rows and columns:", df.shape)
    print(df.head(5))

    # split the data into training and test sets
    # IBM dataset is 1259 lines, which is 1258 data points, so 70% of that is 880
    # we are grabbing a vector that is 880 elements long for the training set
    # everything else is test set (1258-880) = 378
    # we will use open price which is index 1 in the data. or maybe it's 2. im not sure if python counts from 0.
    # in any case the print shows that the open price is what we grabbed.
    training_set = df.iloc[:880, 1:2].values
    test_set = df.iloc[880:, 1:2].values


    # Build the input features with a lag of 1 day.
    # Feature Scaling
    # we scale the price between 0 and 1 for every point in the training set
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 80 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(80, 880):
        X_train.append(training_set_scaled[i-80:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print("X_train.shape is:", X_train.shape)
    # the data has now been reshaped into the format: (#values, #time-steps, #1 dimensional output)
    # #values is equal to the size minus the time steps because every time you make a time step you lose 1 data point

    # build the LSTM.
    # chose 40 neurons and 4 hidden layers
    # assign 1 neuron in the output layer for predicting the normalized stock price
    # use the MSE loss function and the Adam stochastic gradient descent optimizer

    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 40, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 40, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 40, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 40))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))

    # Compiling the RNN
    # loss = error
    # optimizer = gradient descent
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    # Hyperparameters!
    model.fit(X_train, y_train, epochs = 50, batch_size = 30)

    # Prepare the test data (reshape it)
    # Getting the predicted stock price of 2017
    dataset_train = df.iloc[:880, 1:2]
    dataset_test = df.iloc[880:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 80:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []

    # in the loop
    # the first number is the number of time-steps
    # the second number is the size of the test array + time steps
    # the size of the test array would have been the number of data pts (number of rows minus the header)
    # minus whatever you used for the training.
    for i in range(80, 458):
        X_test.append(inputs[i-80:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print("X_test.shape is:", X_test.shape)

    # Use the test data to make predictions for the dates that ... are the test data dates.
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    np.set_printoptions(threshold=np.inf)
    # print(predicted_stock_price)
    np.savetxt('output.out', predicted_stock_price, delimiter=',')
    # ^ the output is normalized! Scaled to 1. So the real predicted prices would be the stuff in output.out
    # multiplied by the max open price in the training data

    # draw a graph to compare
    plt.plot(df.loc[880:, 'Date'], dataset_test.values, color = 'red', label = 'Real Price')
    plt.plot(df.loc[880:, 'Date'], predicted_stock_price, color = 'blue', label = 'Predicted Price')
    plt.xticks(np.arange(0,378,70))
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
