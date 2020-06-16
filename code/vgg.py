# this is used to trian the vgg model to classify panel and nopanel
import keras
import numpy
import os
import sys
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import keras
from keras import models
from keras import layers
from keras.callbacks import TensorBoard
from keras.applications import VGG16
import datetime
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
# data to train vgg model

# Input dirs
train_dir = ' '

validation_dir = ' '


# Output dirs

model_output_dir = ' '

if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)
training_log_dir = model_output_dir
training_model_output_dir = model_output_dir



wp = ' '
# pretrained model imagenet
conv_base = VGG16(weights=wp,
                  include_top=False,
                  input_shape=(256, 256, 3))

vgg16_base = VGG16(include_top=False, weights='imagenet',
                   input_tensor=None, input_shape=(256, 256, 3))

# In[10]:

print('Adding new layers...')
output = vgg16_base.get_layer(index = -1).output
output = Flatten()(output)
# let's add a fully-connected layer
output = Dense(1024,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(1024,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(6, activation='softmax')(output)
print('New layers added!')

vgg16_model = Model(vgg16_base.input, output)
for layer in vgg16_model.layers[:-7]:
    layer.trainable = False
vgg16_model.summary()

conv_base.trainable = True

set_trainable = False

print('trainable weights is :', len(vgg16_model.trainable_weights))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')


# model.compile(loss='binary_crossentropy',  optimizer=optimizers.RMSprop(lr=2e-5), )
vgg16_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
# use checkpointer to stop trainnig early
checkpointer = ModelCheckpoint(filepath = training_model_output_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=200, verbose=1)
log_dir = training_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print (log_dir)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
callbacks = [checkpointer,earlystopper,tensorboard_callback]

history = vgg16_model.fit_generator(
    train_generator,
    samples_per_epoch=1000,
    epochs=300,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2,
    callbacks=callbacks)
path = model_output_dir
vgg16_model.save(os.path.join(path ,'VGG16_pretrain_all.model'))

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

# In[16]:

fig = plt.figure(figsize=(20,10))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.show()
fig.savefig(model_output_dir + 'Accuracy_curve_vgg16.jpg')

# In[17]:

fig2 = plt.figure(figsize=(20,10))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')
fig2.savefig(model_output_dir + 'Loss_curve_vgg16.jpg')

print('finish')

sys.stdout.flush()






