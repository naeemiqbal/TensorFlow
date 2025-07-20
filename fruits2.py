import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import  Dense,GlobalAveragePooling2D


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

src_train = '../data/fruits/dataset/train'
src_test = '../data/fruits/dataset/test'
sub_classes = os.listdir(os.path.join(src_train))
print("Sub-classes found:", sub_classes)
N_CLASSES = len(sub_classes)
classes = []
j=0
for sub_class in sub_classes:
    sub_class_dir = os.path.join(src_train, sub_class) 
    classes.append(sub_class)
    if os.path.isdir(sub_class_dir):
        images = os.listdir(sub_class_dir)
        print(f"Sub-class {sub_class} has {len(images)} images.")
        for i, image in enumerate(images[:5]):  # Display first 5 images
            img_path = os.path.join(sub_class_dir, image)
            img = plt.imread(img_path)
            plt.subplot(6, 5, j*5+ i + 1)
            plt.imshow(img)
            plt.title(sub_class)
            plt.axis('off')
    j += 1
plt.tight_layout()
plt.show()
print("Classes:", classes)


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)

batch_size = 8
train_generator = train_datagen.flow_from_directory(
    directory=src_train,
    target_size=(100,100),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

valid_generator = train_datagen.flow_from_directory(
    directory=src_train,
    target_size=(100,100),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    directory=src_test,
    target_size=(100,100),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    color_mode='rgb',
    seed=42
)
""""
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
"""

base_model = VGG16(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit (train_generator,
           validation_data=valid_generator,
              steps_per_epoch=train_generator.n // train_generator.batch_size,
                validation_steps=valid_generator.n // valid_generator.batch_size,
                epochs=25)
model.save('fruits2_model.keras')