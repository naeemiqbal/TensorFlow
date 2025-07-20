import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import os

model = tf.keras.models.load_model('fruits_model.keras')
classes=['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
#test_dir = os.path.join('../data/fruits/test')
test_dir = os.path.join('../data/fruits/dataset/test/rottenapples')
corr=3
if os.path.isdir(test_dir):
    images = os.listdir(test_dir)
    l = len(images)
    siz = int(np.ceil(np.sqrt(l)))  # Use ceil to ensure enough rows/cols for plotting
    print("Square root of l (rounded up):", siz)
    print(f"Number of images in {test_dir}: {l}")
    for i, image in enumerate(images):
        img_path = os.path.join(test_dir, image)
        img = Image.open(img_path).convert('RGB').resize((100, 100))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 100, 100, 3)
        pred = model.predict(img_array)
        predicted_class = np.argmax(pred)
        print(f"Predicted index:", predicted_class, "Name:", classes[predicted_class])    
        plt.subplot(siz ,siz, i + 1)
        plt.imshow(img)
        plt.title(f"{i+1} {classes[predicted_class]}")
        plt.axis('off')  
        if (predicted_class == 3):
            corr += 1
plt.tight_layout()
plt.show()  

l = len(images)
print(f"Total images: {l}")
print(f"Correct predictions: {corr}")
if l > 0:
    print(f"Accuracy: {corr}/{l} = {corr/l*100:.2f}%")
    