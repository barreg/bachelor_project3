#!/usr/bin/env python
# coding: utf-8

# define the model (classic binary classifier)

# In[12]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Flatten

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.summary()


# define the image generators using keras preprocessing 

# In[30]:


import os 
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                zoom_range=0.2, fill_mode='nearest')


# In[7]:


batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory("board data/final/training", target_size=(224,224), 
                                                   batch_size=batch_size, class_mode="binary")

validation_generator = test_datagen.flow_from_directory("board data/final/validation", target_size=(224, 224), 
                                                       batch_size = batch_size, class_mode = "binary")


# fit the model 

# In[ ]:


history = model.fit_generator(train_generator, steps_per_epoch=2000 // batch_size, epochs=15, validation_data=validation_generator, 
                   validation_steps=800 // batch_size, callbacks = [plot])


# In[31]:


from IPython.display import clear_output
class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()

plot = TrainingPlot()


# save the weights

# In[6]:


import h5py 
model.save_weights('first_try.h5')


# test the model

# In[37]:


#test_generator = test_datagen.flow_from_directory("board data/testing", target_size=(150,150), 
 #                                                  batch_size=batch_size, class_mode="binary")

#model.predict_generator(test_generator, steps = len(test_generator))

