
import tensorflow as tf
import numpy as np

fashion_mnist_data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

train_images = train_images/255
test_images = test_images/255


print(train_images.shape, train_labels.shape)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(10, activation='softmax')])

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(train_images[..., np.newaxis], train_labels, batch_size=256, epochs=2)


