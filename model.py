# Import Packages
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Generate batches of data
def generator(entries, batch_size):
    comp = np.array([0.2, 0.3, 0.4])    # Steering angle compensations
    num_entries = len(entries)
    while 1:                            # Loop forever while training
        entries = shuffle(entries)      # Shuffle driving log entries before each epoch
        for offset in range(0, num_entries, batch_size):
            batch_entries = entries[offset:offset + batch_size]
            images = []
            angles = []
            for batch_entry in batch_entries:
                for i in range(3): # Load camera images
                    image = cv2.imread('data/IMG/' + batch_entry[i].split('/')[-1])
                    angle = batch_entry[3].astype(float)
                    
                    if i == 0:                      # Center camera image steering angle
                        angle = angle
                        
                    elif i == 1:                    # Left camera image steering angle
                        if angle == 0:              # Straight angle compensation
                            angle = comp[0]
                        elif angle < 0:             # Inside left turn angle compensation
                            angle = angle + comp[1]
                        elif angle > 0:             # Outside right turn angle compensation
                            angle = angle + comp[2]
                        
                    elif i == 2:                    # Right camera image steering angle
                        if angle == 0:              # Straight angle compensation
                            angle = -comp[0]
                        elif angle < 0:             # Outside left turn angle compensation
                            angle = angle - comp[2]
                        elif angle > 0:             # Inside right turn angle compensation
                            angle = angle - comp[1]

                    images.append(image)
                    angles.append(angle)

            X = np.array(images)
            y = np.array(angles)
            
            yield shuffle(X, y)
            
# Load driving log entries
entries = []
with open('data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        entries.append(line)
entries = np.array(entries)

# Subsample dataset
steering_angles = entries[:, 3].astype(float)
hist, _ = np.histogram(steering_angles, bins = [-1.1, -0.0001, 0.0001, 1.1]) # Steering angle histogram

per_zeros_del = 0.5 # Percent of examples with a steering angle of zero to delete
num_zeros_del = np.array(hist[1]*per_zeros_del).astype(int) # Number of examples with a steering angle of zero to delete
# Indices of examples with a steering angle of zero to delete
ind_zeros_del = np.random.choice(np.argwhere(np.absolute(steering_angles) < 0.0001).flatten(),\
                                 size = num_zeros_del, replace = False)
entries = np.delete(entries, ind_zeros_del, axis = 0)

steering_angles_new = entries[:, 3].astype(float)
hist_new, _ = np.histogram(steering_angles_new, bins = [-1.1, -0.0001, 0.0001, 1.1]) # Steering angle histogram

entries = shuffle(entries)                                  # Shuffle dataset
subset, _ = train_test_split(entries, train_size = 0.1)     # Create a small subset of dataset
train, valid = train_test_split(subset, train_size = 0.8)   # Split subset into training and validation datasets

# Dataset summary
image = cv2.imread('data/IMG/' + entries[0][0].split('/')[-1])
image_shape = np.array(image).shape

print('Dataset distribution before subsampling straight ahead examples')
print('Number of left turn examples = ', hist[0]*3) # Multiply by 3 to account for left, right, and center images
print('Number of straight ahead examples = ', hist[1]*3)
print('Number of right turn examples = ', hist[2]*3)

print('\nDataset distribution after subsampling straight ahead examples')
print('Number of left turn examples = ', hist_new[0]*3)
print('Number of straight ahead examples = ', hist_new[1]*3)
print('Number of right turn examples = ', hist_new[2]*3)

print('\nNumber of examples after subsampling and data augmentation')
print('Number of training examples = ', len(train)*3)
print('Number of validation examples = ', len(valid)*3)

print('\nInput image shape = ', image_shape)

# Hyperparameters
epochs = 20
batch_size = 32

# Data generators
train_generator = generator(train, batch_size)
valid_generator = generator(valid, batch_size)

# Create model
model = Sequential()

# Image pre-processing
model.add(Lambda(lambda x: (x - 128.0)/128.0, input_shape = image_shape))       # Normalize and zero-mean center images
model.add(Cropping2D(cropping = ((68, 24), (2, 2))))                            # Cropping: Output 68x316x3

# NVIDIA End-to-End model architecture
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))     # Convolution layer 1: Output 32x156x24
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))     # Convolution layer 2: Output 14x76x36
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))     # Convolution layer 3: Output 5x36x48
model.add(Convolution2D(64, 3, 3, activation = 'relu'))                         # Convolution layer 4: Output 3x34x64
model.add(Convolution2D(64, 3, 3, activation = 'relu'))                         # Convolution layer 5: Output 1x32x64
model.add(Flatten())                                                            # Flatten: Output 2048
model.add(Dense(100, activation = 'relu'))                                      # Fully connected layer 1: Output 100
model.add(Dense(50, activation = 'relu'))                                       # Fully connected layer 2: Output 50
model.add(Dense(10, activation = 'relu'))                                       # Fully connected layer 3: Output 10
model.add(Dense(1))                                                             # Output layer

# Optimizer
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer = adam, loss = 'mse')

# Callback that saves only the best epoch
save_model = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, verbose = 2)

# Train model
print('\nTraining...\n')
model.fit_generator(generator = train_generator, samples_per_epoch = len(train)*3,\
                    validation_data = valid_generator, nb_val_samples = len(valid)*3,\
                    callbacks = [save_model], nb_epoch = epochs, verbose = 2)
print('\nTraining Completed')
