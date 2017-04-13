
# coding: utf-8

# In[1]:

# Import Packages
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[2]:

samples = []
with open('./drivesimulator_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[3]:

def generator(samples, correction, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './drivesimulator_data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = './drivesimulator_data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = './drivesimulator_data/IMG/'+batch_sample[2].split('/')[-1]
                
                center_image = cv2.imread(name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                
                images.append(center_image)
                angles.append(center_angle)
                
                images.append(left_image)
                angles.append(left_angle)
                
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print (X_train.shape)
            yield shuffle(X_train, y_train)


# In[4]:

def generator2(samples, correction, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            center_image_old = None
            left_image_old = None
            right_image_old = None
            
            for batch_sample in batch_samples:
                name = './drivesimulator_data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = './drivesimulator_data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = './drivesimulator_data/IMG/'+batch_sample[2].split('/')[-1]
                
                center_image = cv2.imread(name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                
                # cropping the image
                center_image = center_image[50:140, :, :] # crop top and down
                left_image = left_image[50:140, :, :] # crop top and down
                right_image = right_image[50:140, :, :] # crop top and down
                
                
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                
                # increase the feature matrix by merging the current with the previous one
                if center_image_old is not None:
                    center_image_new = np.concatenate((center_image, center_image_old), axis=0)
                    left_image_new = np.concatenate((left_image, left_image_old), axis=0)
                    right_image_new = np.concatenate((right_image, right_image_old), axis=0)
                    
                    #print (center_image.shape)
                
                    images.append(center_image_new)
                    angles.append(center_angle)
                
                    images.append(left_image_new)
                    angles.append(left_angle)
                
                    images.append(right_image_new)
                    angles.append(right_angle)
                
                center_image_old = center_image
                left_image_old = left_image
                right_image_old = right_image

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print (X_train.shape)
            yield shuffle(X_train, y_train)


# In[5]:

image = cv2.imread('./drivesimulator_data/IMG/' + train_samples[0][0].split('/')[-1])
image_shape = np.array(image).shape
print (image_shape)


# In[6]:

# hyper parameter
top = 50
down = 20
left= 0
right = 0
correction = 0.2


# In[16]:

# Callback that saves only the best epoch
save_model = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, verbose = 2)


# In[17]:

# compile and train the model using the generator function
train_generator = generator(train_samples, correction, batch_size=32)
validation_generator = generator(validation_samples, correction, batch_size=32)


# In[18]:

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=image_shape))
model.add(Cropping2D(cropping = ((top, down), (left, right)), input_shape= image_shape)) 
model.add(Dropout(0.8))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu')) # Convolution layer 1: Output 32x156x24
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))     # Convolution layer 2: Output 14x76x36
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))     # Convolution layer 3: Output 5x36x48
model.add(Convolution2D(64, 3, 3, activation = 'relu'))                         # Convolution layer 4: Output 3x34x64
model.add(Convolution2D(64, 3, 3, activation = 'relu'))                         # Convolution layer 5: Output 1x32x64
model.add(Flatten())                                                            # Flatten: Output 2048
model.add(Dense(100, activation = 'relu'))                                      # Fully connected layer 1: Output 100
model.add(Dense(50, activation = 'relu'))                                       # Fully connected layer 2: Output 50
model.add(Dense(10, activation = 'relu'))                                       # Fully connected layer 3: Output 10
model.add(Dense(1))  

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), callbacks = [save_model], nb_epoch=3)


# In[38]:

# Callback that saves only the best epoch
save_model = ModelCheckpoint('model2.h5', monitor = 'val_loss', save_best_only = True, verbose = 2)


# In[39]:

# compile and train the model using the generator function
train_generator = generator2(train_samples, correction, batch_size=32)
validation_generator = generator2(validation_samples, correction, batch_size=32)


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=[180, 320, 3]))
#model.add(Cropping2D(cropping = ((top, down), (left, right)), input_shape= image_shape)) 
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))     # Convolution layer 1: Output 32x156x24
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))     # Convolution layer 2: Output 14x76x36
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))     # Convolution layer 3: Output 5x36x48
model.add(Convolution2D(64, 3, 3, activation = 'relu'))                         # Convolution layer 4: Output 3x34x64
model.add(Convolution2D(64, 3, 3, activation = 'relu'))                         # Convolution layer 5: Output 1x32x64
model.add(Flatten())                                                            # Flatten: Output 2048
model.add(Dense(100, activation = 'relu'))                                      # Fully connected layer 1: Output 100
model.add(Dense(50, activation = 'relu'))                                       # Fully connected layer 2: Output 50
model.add(Dense(10, activation = 'relu'))                                       # Fully connected layer 3: Output 10
model.add(Dense(1))  

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), callbacks = [save_model], nb_epoch=10)


# In[28]:




# In[ ]:



