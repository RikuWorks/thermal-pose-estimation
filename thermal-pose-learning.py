import os
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint


FTRAIN = 'train.csv'
FIMAGE = 'image.csv'

def traindataloader():
    
    df = read_csv(os.path.expanduser(FTRAIN))
    di = read_csv(os.path.expanduser(FIMAGE))
    df = pd.concat([df,di],axis=1)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    print(df.count()) 
    df = df.fillna(method='bfill')
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48 
    X, y = shuffle(X, y, random_state=42)
    y = y.astype(np.float32)
    return X, y

def plot_sample(x, y, axis):
    img = x.reshape(96, 96) 
    axis.imshow(img, cmap='gray') 
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def traindataloader2D():
    X, y = traindataloader()
    X = X.reshape(-1,96, 96,1)
    return X, y

X, y = traindataloader2D()

print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

model = Sequential()

model.add(Conv2D(32, 3, input_shape=(96, 96, 1)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, 2))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, 2))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(36))

os.makedirs('models', exist_ok=True)

model_checkpoint = ModelCheckpoint(

    filepath=os.path.join('models', 'model_{epoch:02d}_{val_loss:.2f}.h5'),

    monitor='val_loss',

    verbose=1,
    period=50)


sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
hist = model.fit(X, y, epochs=10000, validation_split=0.23,callbacks=[model_checkpoint])

model.save('model.h5')
