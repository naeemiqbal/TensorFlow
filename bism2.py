import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
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
model.fit(x_train, y_train, epochs=10)
r = model.evaluate(x_test, y_test)
print("Test loss:", r[0])
print("Test accuracy:", r[1])
# Save the model
model.save('Bism2.keras')

# ...existing code...

import numpy as np
from PIL import Image
# Load and preprocess your image (replace 'your_image.png' with your file)
img = Image.open('test1.png').convert('L').resize((28, 28))
import matplotlib.pyplot as plt 
plt.imshow(img, cmap='gray')
plt.imshow(img)
plt.show()

img_array = 1.0 - (np.array(img) / 255.0)
img_array = img_array.reshape(1, 28, 28)  # Add batch dimension

# Predictea
pred = model.predict(img_array)
predicted_class = np.argmax(pred)
print("Predicted digit:", predicted_class)
# ...existing code...