import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train / 255
X_test = X_test / 255


def get_accuracy(model, X_test, y_test):
    """
    :rtype: accuracy 
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('accuracy: {acc: 0.3f}'.format(acc=test_acc))



def get_model():
    model = Sequential([Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3,3),
                           activation='relu'),
                    Conv2D(filters=8, kernel_size=(3,3), activation='relu'),
                    MaxPooling2D((4, 4)),
                    Flatten(),
                    BatchNormalization(),
                    Dense(32, activation='relu'),
                    Dense(10, activation='softmax')])
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    return model

model = get_model()
get_accuracy(model, X_test, y_test)
print(model.summary())


checkpoint_path = os.path.join('model_checkpoints', 'checkpoints')
checkpoints = ModelCheckpoint(filepath=checkpoint_path,
                              save_freq='epoch',
                              save_best_only=True,
                              monitor='val_loss',
                              mode='min',
                              save_weights_only=True,
                              verbose=1)

def train_model(model):
    model.fit(X_train, y_train, epochs=3, batch_size=64,
          callbacks=[EarlyStopping(patience=20), checkpoints],
          validation_split=0.2)

train_model(model)

get_accuracy(model, X_test, y_test)

model = get_model()
model.load_weights(checkpoint_path)
get_accuracy(model, X_test, y_test)


