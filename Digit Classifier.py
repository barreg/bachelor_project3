#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Flatten, Input, BatchNormalization, ZeroPadding2D, AveragePooling2D


visible = Input(shape=(224, 224, 3))

conv1 = Conv2D(16, (3,3), padding='same', kernel_initializer = 'normal', activation='relu')(visible)
normal1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2,2))(normal1)
conv2 = Conv2D(32, (3,3), padding='same', kernel_initializer = 'normal', activation='relu')(pool1)
normal2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(normal2)

conv3 = Conv2D(16, (1,1), kernel_initializer = 'normal', activation='relu')(pool2)
normal3 = BatchNormalization()(conv3)
conv4 = Conv2D(128, (3,3), padding='same', kernel_initializer = 'normal', activation='relu')(normal3)
normal4 = BatchNormalization()(conv4)
conv5 = Conv2D(16, (1,1), kernel_initializer = 'normal', activation='relu')(normal4)
normal5 = BatchNormalization()(conv5)

conv6 = Conv2D(128, (3,3), padding='same', kernel_initializer = 'normal', activation='relu')(normal5)
normal6 = BatchNormalization()(conv6)
pool6 = MaxPooling2D(pool_size=(2,2))(normal6)

conv7 = Conv2D(32, (1,1), kernel_initializer = 'normal', activation='relu')(pool6)
normal7 = BatchNormalization()(conv7)
conv8 = Conv2D(256, (3,3),padding='same', kernel_initializer = 'normal', activation='relu')(normal7)
normal8 = BatchNormalization()(conv8)
conv9 = Conv2D(32, (1,1), kernel_initializer = 'normal', activation='relu')(normal8)
normal9 = BatchNormalization()(conv9)
conv10 = Conv2D(256, (3,3), padding='same', kernel_initializer = 'normal', activation='relu')(normal9)
normal10 = BatchNormalization()(conv10)
pool8 = MaxPooling2D(pool_size=(4,4), strides=(4,4))(normal10)

avg = AveragePooling2D((7,7))(pool8)

output1 = Conv2D(11, (1,1), kernel_initializer = 'normal', activation = 'softmax')(avg)
output2 = Conv2D(11, (1,1),  kernel_initializer = 'normal', activation = 'softmax')(avg)
output3 = Conv2D(11, (1,1), kernel_initializer = 'normal', activation = 'softmax')(avg)
output4 = Conv2D(11, (1,1), kernel_initializer = 'normal', activation = 'softmax')(avg)

model = Model(inputs = visible, outputs = [output1, output2, output3, output4])

model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy','categorical_crossentropy'], metrics=['accuracy'])
model.summary()


# In[2]:


import os

def load_paths(path):
    paths = []
    for file in os.listdir(path):
        if file == '.DS_Store':
            continue
        paths += [path +'/'+ file]
    return paths


# In[3]:


training_data_paths = load_paths('digits data/boards held4/training')
validation_data_paths = load_paths('digits data/boards held4/validation')


# In[4]:


import cv2 
import numpy as np

def get_input(path):
    img = cv2.imread(path)
    return img 


# In[5]:


import torchvision
import torchvision.transforms.functional as transforms
from PIL import Image

def preprocess_input(img):
    img = Image.fromarray(img)
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    #brightness, rotate, shear of random values scale 
    brightness = np.random.uniform(0.5, 1.2)
    angle = np.random.randint(-20,20)
    scale = np.random.uniform(2)
    shear = np.random.uniform(-20,20)
    translate = (0,0)

    new_img = torchvision.transforms.functional.affine(img, angle, translate, scale, shear, fillcolor=0)
    new_img = torchvision.transforms.functional.adjust_brightness(new_img, brightness)
    new_img = np.array(new_img)
    return new_img


# In[6]:


def get_output(path):
    output1 = np.zeros((1,1,11))
    output2 = np.zeros((1,1,11))
    output3 = np.zeros((1,1,11))
    output4 = np.zeros((1,1,11))
    
    img_id = path.split('-')[-1].split('.')[0]
    label = np.load('digits data/labels2/label_'+img_id+'.npy')
    
    output1[0][0][label[0]] = 1
    output2[0][0][label[1]] = 1
    output3[0][0][label[2]] = 1
    output4[0][0][label[3]] = 1
    
    return output1, output2, output3, output4


# In[7]:


def image_generator(files, batch_size):
    
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a = files, 
                                         size = batch_size)
        batch_input = []
        batch_output1 = [] 
        batch_output2 = []
        batch_output3 = []
        batch_output4 = []
          
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            img = get_input(input_path)
            batch_output = get_output(input_path)

            new_img = preprocess_input(img) 
            batch_input += [ new_img ]
            batch_output1 += [ batch_output[0] ]
            batch_output2 += [ batch_output[1] ]
            batch_output3 += [ batch_output[2] ]
            batch_output4 += [ batch_output[3] ]
            
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y1 = np.array( batch_output1 )
        batch_y2 = np.array( batch_output2 )
        batch_y3 = np.array( batch_output3 )
        batch_y4 = np.array( batch_output4 )
        
        yield( batch_x, [batch_y1, batch_y2, batch_y3, batch_y4])


# In[8]:


batch_size = 32

training_gen = image_generator(training_data_paths, batch_size)
validation_gen = image_generator(validation_data_paths, batch_size)


# In[30]:


model.fit_generator(training_gen, steps_per_epoch= 125, epochs=15, validation_data=validation_gen, 
                   validation_steps= 125)


# In[9]:


import h5py 
model.save_weights('speedy_firt.h5')


# In[45]:


#tests = []
#test = cv2.imread('digits data/test/data_5-11209 copie.jpg')
#tests += [test]
#tests += [test]
#final_test = np.array(tests)
#model.predict(final_test)


# In[43]:


#print(get_output('digits data/boards held4/training/data_73471-4862.jpg'))


# In[78]:


#import cv2 
#import matplotlib.pyplot as plt
#for i, path in enumerate(validation_img_paths) : 
    
    #rd = np.random.randint(500, 4500)
    #if(i<5):
        #img = get_input(path)
        #img = preprocess_input(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       # plt.imshow(img)
       # out = get_output(path)
        #print(out)


# In[17]:


#out = get_output('digits data/boards held/training/data_'+str(rd)+'.jpg')
#print(out)


# In[18]:


#img = get_input('digits data/boards held4/training/data_4-12700.jpg')
#print(img.size)


# In[27]:


#import cv2 
#import matplotlib.pyplot as plt 
#import numpy as np


##print(get_output('digits data/boards2/training/board_12843.jpg')[0][0][0])


# In[ ]:


#questions : 
   # in the get_output does the 4 np.zeros(11) make the thing longer ? allocate memory elsewhere ???
   # where to do the maxpool ? how many ?
   # max number of parameters ? 
    
    
   # 224 x 224 
    
   # 4 max pools
    
   # ref_11 OK 
    


# In[20]:


#img = preprocess_input(get_input('digits data/resized refs/ref_10.jpg'))

#img2 = Image.fromarray(img)
#img2.show()


# In[5]:


#img = get_input('digits data/resized refs/ref_10.jpg')
#img2 = Image.fromarray(img)
#b, g, r = img2.split()
#img2 = Image.merge("RGB", (r, g, b))
#img2.show()


# In[56]:


#import torchvision
#import torchvision.transforms.functional as transforms
#import numpy as np
#from PIL import Image

#img  = Image.open("digits data/boards held4/training/data_48319-4090.jpg")

#b, g, r = img.split()
#img = Image.merge("RGB", (r, g, b))
#brightness, rotate, shear of random values scale 
#brightness = np.random.uniform(0.5, 1.2)
#angle = np.random.randint(-20,20)
#scale = np.random.uniform(2)
#shear = np.random.uniform(-20,20)
#translate = (0,0)

#new_img = torchvision.transforms.functional.affine(img, angle, translate, scale, shear, fillcolor=0)
#new_img = torchvision.transforms.functional.adjust_brightness(new_img, 1.2)
#new_img = np.array(new_img)
#new_img.show()


# In[52]:


#for i in range (100):
  #  print(np.random.uniform(2))


# In[ ]:


#move my data to speedy and jupyter notebook to python file then train 


# In[ ]:


#sshfs -o IdentityFile=~/.ssh/id_rsa -p 37559 guillaume@62.2.206.134:/ speedy_mounted

