import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time


NAME = "Cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sses = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# model.add(Dense(64))
# model.add(Activation('relu'))

model.add(Dense(1)) 
model.add(Activation("sigmoid")) # softmax

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) 

model.fit(  X, y, 
            batch_size=32,
            validation_split=0.3,
            epochs=20,
            callbacks=[tensorboard]
        )             

score, acc = model.evaluate(X, y, verbose=0)
print('\n')
print('Test score:', score)
print('Test accuracy:', acc)

#     "from tensorflow.keras.wrappers.scikit_learn import KerasClassifier\n",
#     "\n",
#     "estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)\n",
#     "nn_scores = cross_val_score(estimator, all_features_scaled, all_classes, cv=10)\n",
#     "nn_scores.mean()\n",        



#     "clf.predict(test_features_trees)\n",
#     "\n",
#     "# clf.predict_proba(test_features_trees)\n",
#     "clf.score(test_features_trees, test_classes_trees)\n",