import glob
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
np.random.seed(42)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#get dataset
   
IMG_DIM = (150, 150)

train_files = glob.glob('training_data/*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in train_files]

validation_files = glob.glob('validation_data/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in validation_files]

print('Train dataset shape:', train_imgs.shape, 
      '\tValidation dataset shape:', validation_imgs.shape)

#scale img pixels
train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled  = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape)
array_to_img(train_imgs[0])
#==================================================================

#Basic parameters
batch_size = 30
num_classes = 2
epochs = 30
input_shape = (150, 150, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)

print(train_labels[1495:1505], train_labels_enc[1495:1505])

#===================================================================

#get pre trained VGG16 network
from keras.applications import vgg16
from keras.models import Model
import keras


vgg = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = input_shape)
output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input , output)
#model ready

#Froze he model
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

#All the layers and the model is frozen now
#==========================================================

#Defining our model
from keras.layers import Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation = 'relu', input_dim = input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4), metrics = ['accuracy'])

#===============================================================

#Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( rescale=1./255, rotation_range= 50,
                                    shear_range=0.2,
                                    zoom_range=0.3,
                                    horizontal_flip=True )

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size = 30)

val_generator = train_datagen.flow(validation_imgs, validation_labels_enc, batch_size = 20)


#===================================================================

#As data and model is ready.........training starts

hist = model.fit_generator(train_generator, steps_per_epoch = 100,
                           epochs= 25, validation_data = val_generator,
                           validation_steps = 50, verbose=1)


