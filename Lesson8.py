import pandas as pd
import os
import time
import random
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# The end goal is to show any 60 minutes of data to the model and it will tell if the price is going up or down. 
# If we show the data in order it doesn't learn much as 58 minutes of the data is the same as the previous iteration. 
# Shuffling makes sure that the situation should be completely random which it will be when the model is deployed in the future.
 


SEQ_LEN = 60 # X ostatnich minut, na podstawie której będziemy przewidywać
FUTURE_PERIOD_PREDICT = 3 # X min które chcemy przewidzieć
RATIO_TO_PREDICT = "LTC-USD" # którą walutę chcemy przewidzieć
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

# df = pd.read_csv("crypto_data/LTC-USD.csv", names=["time", "low", "hight", "open", "close", "volume"])
# print(df.head())

# Jeśli cena w przyszłości jest większa niż obecna cena,
# w tedy powiemy 1 w innym przypadku zwrócimy 0
# 1 oznacza "kupuj", 0 oznacza "nie kupuj"
# Chcemy nauczyć sieć aby na podstawie sekwencji atrybutów zachowywała się w pewien sposób
# czyli np. jeżeli w pierwszej minucie cena idzie do góry, a w drugiej w dół i w trzeciej minucie
# także w dół to kupuj
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

# Tworzymy tablicę tablic (w pythonie listę list) z 60cio minutowymi interwałami cen kryptowalut
# [[60min], [60min], [60min], [60min], ..., [60min]] - gdzie [60min] składa się z [[np(atrybuty), target]]
# Na tej podsatwie będziemy określać na podstawie nowych 60 minut z cenami, czy cena pójdzie do góry czy spadnie
# W tej funkcji dochodzi do normalizacji danych i skalowaniu danych
def preprocess_df(df): #df - data frame
    df = df.drop('future', 1) # pozbywamy się kolumny future, aby sieć tego nie zapmiętała

    for col in df.columns:
        if col != "target": # kolumna target jest już znormalizowana więc ją pomijamy
            # dla pozostałych kolumn normalizujemy dane (np. btc, eth, itp.)
            # czyli zamieniamy ceny na wachania procentowe (jeżeli dobrze to rozumiem?)
            # bo każda waluta ma inny przelicznik cen
            df[col] = df[col].pct_change()
            # pozbywamy się wartości N/A, w pct_change czasami mogą się tam pojawić
            df.dropna(inplace=True) 
            # skalujemy wartości, sklearn ma dobrą funkcję pod to (w keras też jest podobna)
            # dzięki czemu uzyskujemy dane które są pomiędzy 0 a 1
            df[col] = preprocessing.scale(df[col].values) # można też preprocessing.MinMaxScaler()  scaler.fit_transform(df[colsToScale])
    # pozbywamy się jakiś wartości N/A jeszcze raz, tak na wszelki wypadek,
    # tak aby mieć pewność że zawsze mamy wartości liczbowe a nie N/A
    df.dropna(inplace=True)

    sequential_data = [] # tworzymy pustą tablice/listę, której będziemy przetrzymywać sekwencję danych
    # Można wyobrazić sobie deque jako listę
    # gdzie itemy zostają dodane, ale gdy dojdzie do wartości maxlen wywala z listy poprzednią/ostatnią wartość
    # to jest nasza sekwencja
    prev_days = deque(maxlen=SEQ_LEN) # tak jakby inicjalizacja obiektu w PHP
    
    # Konwertujemy DataFrame na listę list, gdzie pozbywamy się kolumny time i target
    # i tworzymy sobie sekwencje danych, po 60 min (SEQ_LEN), w każdym w "koszyczku"
    for i in df.values: # i - to jest nasz row, z cenami krypto i kolumną target
        # wrzuca do prev_days po kolei każdy z wierszy z df.values czyli i bez kolumny target
        prev_days.append([n for n in i[:-1]])
        
        # tworzyli pod listę/tablice z danymi z ostatnich 60ciu minut (SEQ_LEN minut)
        # Po to aby później w modelu na podstawie tych 60 minut przewidywać czy cena spadne czy wzrośnie
        # na podstawie przedziału 60 nowych minut
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]]) # prev_days - featury; labele - i[-1]

    # mieszamy randomowo listy z sekwencjami danych
    random.shuffle(sequential_data)

    # Lesson 10

    # Teraz trzeba napisać funkcję która będzie balansować danycmi
    # Jeżeli mamy dane np. podzielone na 48 i 52% to teoretycznie nie potrzeba balansować danymi
    # Gdy mamy 60 i 40%, to już warto balansować danymi
    # Dobre praktyki mówią że lepiej jest balansować daymi niż tego nie robić
    # Po to się to robi aby model nie utchnął podczas nauki
    # Podobny case był ze zdjęciami kotów i psów
    # Najlepsze wyniki daję nam posiadanie 50% zdjęć psów i 50% zdjęć kotów
    buys = []
    sells = []
    
    # na podstawie targetu tworzymy dwie tablice,
    # tablicę mówiącą kiedy kupić, a kiedy sprzedać
    # print(sequential_data[1])
    for seq, target, in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    
    
    # teoretycznie nie potrzeba mieszać danych bo robiliśmy to wyżej, ale dla pewności warto
    random.shuffle(buys)
    random.shuffle(sells)

    # weź tą grupę w której jest mniej danych - lower zlicza ile jest el. mniejszej tablicy
    lower = min(len(buys), len(sells))

    # aby mieć dane 50:50%, okrój tablice buys i sells do wilekości mniejszej tablicy
    # czyli, np. jeżeli w buys jest 1002 el. a w sells więcej 2051 to weź całe buys (bo ma 1002)
    # oraz 1002 el. z sells, dzięki czemu mamy obie tablice takie same po 1002 el.
    buys = buys[:lower]
    sells = sells[:lower]

    # nadpisujemy sequntial_data, nowymi podtablicami buys i sells które mają po tyle samo el
    # i mieszamu el. aby randomowo sie rozłożyły
    sequential_data = buys+sells
    random.shuffle(sequential_data)

    # jako że model przujmuje atrybuty X i labele y, musimy podzielić tablice na X i y
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    
    # zwracamy atrybuy i labele
    # jako że model przyjmuje dla atrybutów tylko tablicę numpy zrzutujemy ją na np
    # Nie jestem pewien czy nadal musimy tak robić skoro wcześniej zrzucalismy atrybuty do np
    
    # atrubuty
    # X - [[60min], [60min], [60min], ..., [60min]]
    #  [
    #      [-2.65323633e-03 -4.25747980e-02  4.04838613e-03 -1.10310340e-02 4.60111811e-03 -3.40638510e-02  3.10232629e-03 -6.02710395e-03]
    #      [-6.79993124e-01  5.18565940e-02 -3.72622718e-01 -1.07957483e-01 -4.95169106e-01 -4.54458134e-02 -5.64421248e-01 -6.12816799e-03]
    #      ...
    #  ] 

    # labele
    # y - [target, target, target, ..., target]
    #     [0 1 0 0 0 1 1 0 1 0 1 ... 0]
    return np.array(X), y

# Jako że chcemy zemergować dane ze wszystkich plików (kryptowalut) do jednego dataseta,
# a każde z nich ma datę można to zrobić
main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"

    df = pd.read_csv(dataset, names=["time", "low", "hight", "open", "close", "volume"])
    # print(df.head())

    # zmieniamy nazwy kolumn ponieważ mogą się powtarzać dla każdej z kryptowaluty
    # a dzięki inplace nie musimy redefiniować DataFrame
    df.rename(columns={"close" : f"{ratio}_close", "volume" : f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    # print(df.head())

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

# Tworzymy nową kolumnę która
# Dane/Ceny z przyszłości będą podawane na podstawie cen "_close", czyli kwoty "zamknięcia"
# Dla "feature" pobieramy jakby ceny o 3 miejsca wyżej pod względem czasu
# Towrzymy nową kolumnę dla ułatwienia operacji na przyszłych danych (patrz. main_df['target])
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

# time            LTC-USD_close   future
# 1528968660      96.580002       96.500000   =>  96.500000 jest 3 miejsca "wyżej"
# 1528968720      96.660004       96.389999
# 1528968780      96.570000       96.519997
# 1528968840      96.500000       96.440002   => czyli tutaj
# 1528968900      96.389999       96.470001
# print(main_df[[f"{RATIO_TO_PREDICT}_close", "future"]].head(10))

# Tworzymy nową kolumnę gdzie
# wrzucamy do funkcji "classify" parametry "current" i "feature"
# Dzięki czemu pokazuje nam czy przyszła cena jest większa (1) czy mniejsza (0) niż
# niż 3 cykle wcześniej
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df['future']))

# time            LTC-USD_close   future      target
# 1528968660      96.580002       96.500000   0       => przyszła cena jest mniejsza od obecnej, więć jest 0
# print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))

# Lesson9 

# Chcemy mieć pewność że dane są posortowane w odpowiednim porządku względem czasu
times = sorted(main_df.index.values)

# Pobieramy 5 ostatnich procent danych (najpewniej jest to zrobić na kolumnie daty)
last_5pct = times[-int(0.05*len(times))]
# print(last_5pct) # ostatni wpis/wiersz timestampu z tych 5 procent

# Tworzymy tablicę z 5% danych 
validation_main_df = main_df[(main_df.index >= last_5pct)]
# Pobieramy dane które mają 95% danych
main_df = main_df[(main_df.index < last_5pct)]

# preprocess_df(main_df) # TODO: wywalic


# Teraz musimy zrobić
# sequences, balance, scale
#  po to nam funkcje które będą robić te rzeczy
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df) # dane testowe


# Lesson 10
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}") # jest idealnie zbalansowane 50:50
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}") # jest idealnie zbalansowane 50:50



# Lesson 11
model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs_crypto/{NAME}')

# zapisujemy tylko te epoki które są najlepsze
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}" # unikalny plik który będzie zawierał epokę i acc dla tej epoki
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # zapisz tylko najlepsze

history = model.fit(train_x, train_y, 
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(validation_x, validation_y),
                    callbacks=[tensorboard, checkpoint])
