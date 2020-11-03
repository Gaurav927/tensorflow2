from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.regularizers import l1, l1_l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print("Starting training.....")
    def on_train_batch_begin(self, batch, logs=None):
        print(f"Starting batch {batch}")
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

data = load_diabetes()
X = data['data']
y = data['target']
y = (y - y.mean(axis=0))/y.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)




model = Sequential([Dense(128, activation='relu', input_shape=(X_train.shape[1],),
                          kernel_regularizer=l1_l2(l1=0.001, l2=0.002),
                          bias_regularizer=l1(0.001)),
                    Dropout(0.4),
                    BatchNormalization(),
                    Dense(128, activation='relu', kernel_regularizer=l1(0.01)),
                    Dropout(0.4),
                    BatchNormalization(),
                    Dense(64, activation='relu'),
                    Dense(1)])

print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=100,
          validation_split=0.15,
          batch_size=64, callbacks=[TrainingCallback()])

print(model.evaluate(X_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

model.fit(X_train, y_train, epochs=5, callbacks=[my_callback])
