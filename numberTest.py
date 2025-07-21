import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
"""
3a 7/10  [0 1 2 3 4 6 7]
2  0/10
2e15 2/10 [1 7]
3b 6/10 [1 2 3 4 5 7]
3ce50 7/10 [1 2 3 4  5 6 9]
"""

pathName='../models/Number3.keras'
model = tf.keras.models.load_model(pathName)
print("No image inverting applied.")
corr=0
for i in range(10):
    fil=f'../data/numNMI/test{i}.png'
   # fil=f'../data/numNMI/s{i}.png'  
    img = Image.open(fil).convert('L').resize((28, 28))
    img_array = (np.array(img) / 255.0)
    img_array = img_array.reshape(1, 28, 28)  # Add batch 
    pred = model.predict(img_array)
    predicted_class = np.argmax(pred)
    print(f"Input {i} , Predicted digit:", predicted_class) 
    plt.subplot(5,4, i*2 + 1)
    plt.title(f"In:{i}  Pr:{predicted_class}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')    
    plt.subplot(5,4, i*2 + 2)
    plt.axis('off')
    plt.imshow(img)
    if (predicted_class == i):
        corr += 1
        plt.title(f"Correct")
#plt.tight_layout()    
plt.show()
print(f"Accuracy: {corr}/10 = {corr*10}%   Using model {pathName}" )