import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# print(x_train.shape)
# print(x_train[0].shape)

x_train = x_train/255.0
x_test = x_test/255.0

model = Sequential()

#CuDNNLSTM - jest o wiele szybszy (ok 5x) od LSTM
# return_sequences=True - oznacza że zwróci nam dane do rekurencyjnej warstwy z zachowaniem sekwencji
model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True)) #x_train.shape[1:] == 28x28
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))# usuwamy activation='relu') bo używamy CuDNNLSTM
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

# robimy mniejsze "skoki/kroki" aby znaleźć minimum lokalne
# gdybyśmy robili większe, moblibyśmy przeskoczyć "dołek" w paraboli
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

# mean_squared_error = mse (skrót)
# sparse_categorical_crossentropy - jest innym sposobem mierzenia zakresu błędu
# inne skróty KLD, MAE, MAPE, MSE, MSLE
model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# czerwona flaga gdy !
# val_acc jest większę niż acc
# oznacza to że model nadal powinno się trenować