##Regression
#Purpose is to predict the values of continuous outputs like prices or probabilities

from google.colab import drive
drive.mount('/content/gdrive')

!pip install -q seaborn

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, 
                          na_values = "?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

dataset.isna().sum()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japen'] = (origin == 3)*1.0
dataset.tail()

##Divide dataset into trainset and testset
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

##Investigate data
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
              diag_kind='kde')

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

##Seperate target value or "label" from features
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

##Data Normalization
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

##Make a model with two densely connected layers
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

model.summary()

##Conduct the model once.
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

##Train the model with 1,000 epoch
##The object of history records train accuracy and validation accuracy
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    '''Express the process of training
       by printing a dot whenever every epoch ends up'''
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

import matplotlib.pyplot as plt

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label='Val Error')
  
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label='Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)

##The result above looks that the model was not improved in spite of progressing many epochs
#By using the method 'model.fit', make it stop automatically if the validation score is not better
#Use 'EarlyStopping'
model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model_check = keras.callbacks.ModelCheckpoint(filepath='/content/gdrive/MyDrive/PRACTICE/Regression_best_model.h5',
                                              monitor='val_loss',
                                              save_best_only=True)

history = model.fit(
    normed_train_data, train_labels, 
    epochs=EPOCHS,
    batch_size = 32,
    validation_split = 0.2, 
    verbose = 0, 
    callbacks=[early_stop, model_check, PrintDot()])

plot_history(history)

##Confirm the performance of the model in testset where data does not be used during the training phase
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Mean Absolut Error:{:5.2f} MPG".format(mae))

##Prediction
#Finally, anticipate the MPG value employing the samples in the testset
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

'''Summary
 1. MSE usually used in the projects of regression
 2. Criteria from regression is also different from that of classification
 3. Have to adjust independently the scale of each feature if they have several ranges
 4. Choose the size of a network that has small counts of densely connected layers if training data is not enough, for avoiding overfitting
 5. Good solution is using 'Early stopping' for averting overfitting'''
