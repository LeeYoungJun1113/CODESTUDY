import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

##This project is related to Overfitting and Underfitting
#For the solution in overfitting, it is dropout and regularization of weights
#Transform this sentence into multi-hot encoding for quickly diving into the overfitting.

NUM_WORDS = 1000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # Make an matric - (len(sequences), dimension) - filled with 0
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # results[i]의 특정 인덱스만 1로 설정합니다
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])

##The best solution about Overfitting is to minimize the size of a model
#By decreasing the number of parameters learnable in models.
#Should be balanced between the too much capabilities and not so enough capabilities
#For finding the best network, Need to many trials and errors.
#Make a baseline model for comparing smaller and bigger model
baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

from keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor="loss", patience=20, mode="min")
tb_hist = TensorBoard(
    log_dir='graph', histogram_freq=0, write_graph=True, write_images=True
)

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2,
                                      callbacks=[tb_hist, early_stopping])

# %load_ext tensorboard
# %tensorboard --logdir {"/content/graph"}

##For comparsion from baseline to smaller to bigger
#Make a smaller model

smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

from keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor="loss", patience=20, mode="min")
tb_hist = TensorBoard(
    log_dir='graph', histogram_freq=0, write_graph=True, write_images=True
)

smaller_history = smaller_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2,
                                      callbacks=[tb_hist, early_stopping])

##Making too much model for overfitting

bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])

bigger_model.summary()

from keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor="loss", patience=20, mode="min")
tb_hist = TensorBoard(
    log_dir='graph', histogram_freq=0, write_graph=True, write_images=True
)

bigger_history = bigger_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2,
                                      callbacks=[tb_hist, early_stopping])

##Plot the graph as to train loss and validation loss

def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')
    
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

plot_history([('baseline', baseline_history),
                ('smaller', smaller_history),
                ('bigger', bigger_history)])

##Regularize weights
#For overfitting, simple model tends to be lower than any complicated model
#In this case, the simple model means model of a small entropy or small parameters
#Use the L2 Regularization known as weight decay
#-->L2 is to limit several parameters to be close to zero, but perfectly not to make to zero like L1 Regularization

l2_model = keras.models.Sequential([
      keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                         activation='relu',
                         input_shape=(NUM_WORDS,)),
      keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                         activation='relu') ,
      keras.layers.Dense(1, activation='sigmoid')
])
##L2(0.001) means that all the values of weight matrix in layers on total loss is
# adding by 0.001 * weight_coefficient_value**2
       
l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2,
                                callbacks=[tb_hist, early_stopping])

##Check the effect of L2 regularization

plot_history([('baseline', baseline_history),
              ('l2', l2_model_history)])

##As shown results above, parameters of each model is the same; However,
#the model applied L2 regularization endured much better than the current model

##Append Dropout one of the affordable solution as to regularization
#While training, the layer mounted a dropout randomly turns off the output features of the layer(that is, make it zero)
#However, no units are dropped out during the test phase.
#Look over how much the overfitting is reduced by affixing two layers of Dropout

dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2,
                                callbacks=[tb_hist, early_stopping])

##As the results shown above, after affixing dropouts in layers, 
#the applied model definitely ameliorated more than before

plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])

##In summary, for preventing from overfitting during the train phase, 
#The most effective way is to follow as:
  #1. Correct train datas more
  #2. Reduce the capability of network
  #3. Accumulate weight regularization
  #4. Append Dropout
