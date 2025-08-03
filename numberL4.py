import tensorflow as tf
print('TensorFlow version - ',tf.__version__)
# Check if GPU is available
gpu_available = tf.config.list_physical_devices('GPU')

if gpu_available:
    print("TensorFlow is installed as GPU version.")
else:
    print("TensorFlow is installed as CPU version.")


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Training data shape:", x_train.shape, y_train.shape)
print("Test data shape:", x_test.shape, y_test.shape)   


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
r= model.evaluate(x_test, y_test)
print("{r}  Test loss:", r[0])
print("Test accuracy:", r[1])
# Save the model
r= model.evaluate(x_test,  y_test, verbose=2)

print("Test loss:", r[0])
print("Test accuracy:", r[1])

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])



model.save('../models/NumberL4.keras')

# Load the model

print("Model training and evaluation completed successfully.")