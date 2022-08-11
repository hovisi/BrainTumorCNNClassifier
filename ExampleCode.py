import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

import sklearn.model_selection as sklearnmodels


def readImageDirectory(basedir, classDict, imageSize=None, quiet=True, testRatio=0.167):
  """
  Take a base directory and a dictionary of class values,
  to into that directory and read all the images there to
  build an image data set.  It is assumed that different
  classes are in different subdirectories with that class
  name.  Return the tensor with the images, as well as the 
  label Y vector.  If a tuple (x,y) imageSize is given,
  then enforce all the images are of a specific size.
  """
  # Initialize the X tensor and the Y vector raw structures
  imagesX = []
  imagesY = []

  for classKey in classDict:
    dirName = os.path.join(basedir, classKey)
    for filename in os.listdir(dirName):
        # Filename name is the base name + class name + image file name
        fn = os.path.join(dirName, filename)

        # If we want to, we can print the images file names as we read them
        if not quiet:
          print(fn)

        # Load the image, then make sure it is scaled all all three color channels
        rawImage = tf.keras.preprocessing.image.load_img(fn, target_size=imageSize )
        image = tf.keras.preprocessing.image.img_to_array(rawImage)/255.0

        # Grow the image tensor and the class vector by 1 entry
        imagesX.append(image)
        imagesY.append(classDict[classKey])

  # Return these as a tensor and a numpy vector
  trainX, testX, trainY, testY = sklearnmodels.train_test_split(imagesX, imagesY, test_size=testRatio)
  return tf.convert_to_tensor(trainX), tf.convert_to_tensor(testX), np.array(trainY, dtype="float32"), np.array(testY, dtype="float32")

  #return tf.convert_to_tensor(imagesX), np.array(imagesY, dtype="float32")


####DATASET USED FOR THIS MODEL : Chakrabati Data#######

## 1) Get the Brain Tumor Data

# Grab all the data from that directory
#From: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
Chakrabarty = "/home/acc.hovisi2/Downloads/CDS"
trainX, testX, trainY, testY = readImageDirectory(Chakrabarty, {"yes":1, "no":0}, (224,224), False )

# Let's setup the Y labels to be compatible with a "hot-ones" representation
trainY = tf.keras.utils.to_categorical(trainY)
testY  = tf.keras.utils.to_categorical(testY)

#print("DBG2: ", trainX.shape)
#print("DBG2: ", trainY.shape)
#print("DBG2: ", testX.shape)
#print("DBG2: ", testY.shape)


### 2) Build the ANN model -- a four-layer CNN

# Start the model
model1 = tf.keras.models.Sequential()

# Add a convolutional layer that takes in tensors 224x224x3
#Then pooling layer 
#repeat for however many convolution layers


model1.add( tf.keras.layers.Conv2D(55,\
                                  (6, 6),\
                                  activation="sigmoid",\
                                  input_shape=(224, 224, 3)) )

model1.add( tf.keras.layers.AveragePooling2D((2, 2)) ) 

model1.add( tf.keras.layers.Conv2D(55,\
                                  (5, 5),\
                                  activation="sigmoid",\
                                  input_shape=(109, 109, 55)) )

model1.add( tf.keras.layers.AveragePooling2D((2, 2)) )                   

model1.add( tf.keras.layers.Conv2D(55,\
                                  (4,4 ),\
                                  activation="sigmoid",\
                                  input_shape=(52, 52, 55)) )

model1.add( tf.keras.layers.AveragePooling2D((2, 2)) )                         
                                  



# Maybe we've got the feature selection we need, so flatten the image
# to a 1D vector.  After this, it's just straight-forward MLP.
model1.add( tf.keras.layers.Flatten() )

# Make a middle layer of 15, each using rectified linear activation
model1.add( tf.keras.layers.Dense(15, activation="relu") )

# Make the output size 2. Let's softmax the activation so
# we get probabilities in the end.
model1.add( tf.keras.layers.Dense(2, activation="softmax") )

# Let's see what the model looks like:
print(model1.summary())

### 3) Train the Model
#Setup categorical crossentropy.  I can do this rather than the
# sparse from-logits because I've reshaped the Y vectors to be hot-ones.
lossFunction = tf.keras.losses.CategoricalCrossentropy()

# Set the optimizer as a stochastic gradient descent method with
# a larning rate of 0.01 
opt = tf.keras.optimizers.SGD(lr=0.01)

# Set the model for training
#  * Use stochastic gradient descent for optimization
#  * Compute loss using sparse categorical cross entropy
#  * Report the performance during learning using accuracy
model1.compile( optimizer=opt, loss=lossFunction, metrics=['accuracy'])

# Set the optimizer as a stochastic gradient descent method with
# a larning rate of 0.01 # Perform the induction
trainingHistory = model1.fit(trainX, trainY, epochs=15)


### 4) Evaluate the Model on the Test Data

# Show the testing accuracy over the un-thresholded model:
print("Evaluate DS1")
model1.evaluate(testX, testY)



print()
print("Starting next dataset: ")
print()
#####DATASET USED FOR THIS MODEL : BTD 2020#######

### 1) Get the Brain Tumor Data

# Grab all the data from that directory
#From: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
Chakrabarty = "/home/acc.hovisi2/Downloads/archive"
trainX, testX, trainY, testY = readImageDirectory(Chakrabarty, {"yes":1, "no":0}, (224,224), False )

# Let's setup the Y labels to be compatible with a "hot-ones" representation
trainY = tf.keras.utils.to_categorical(trainY)
testY  = tf.keras.utils.to_categorical(testY)

### 2) Build the ANN model -- a four-layer CNN
# Start the model
model = tf.keras.models.Sequential()

# Add a convolutional layer that takes in tensors 224x224x3
#Then pooling layer 
#repeat for however many convolution layers

model.add( tf.keras.layers.Conv2D(55,\
                                  (7, 7),\
                                  activation="relu",\
                                  input_shape=(224, 224, 3)) )

model.add( tf.keras.layers.MaxPooling2D((2, 2)) )  

model.add( tf.keras.layers.Conv2D(55,\
                                  (6, 6),\
                                  activation="relu",\
                                  input_shape=(109, 109, 55)) )

model.add( tf.keras.layers.MaxPooling2D((2, 2)) )  



# Maybe we've got the feature selection we need, so flatten the image
# to a 1D vector.  After this, it's just straight-forward MLP.
model.add( tf.keras.layers.Flatten() )

# Make a middle layer of 15, each using rectified linear activation
model.add( tf.keras.layers.Dense(15, activation="relu") )

# Make the output size 2. Let's softmax the activation so
# we get probabilities in the end.
model.add( tf.keras.layers.Dense(2, activation="softmax") )

# Let's see what the model looks like:
print(model.summary())

### 3) Train the Model

# Setup categorical crossentropy.  I can do this rather than the
# sparse from-logits because I've reshaped the Y vectors to be hot-ones.
lossFunction = tf.keras.losses.CategoricalCrossentropy()

# Set the optimizer as a stochastic gradient descent method with
# a larning rate of 0.01 
opt = tf.keras.optimizers.SGD(lr=0.01)

# Set the model for training
#  * Use stochastic gradient descent for optimization
#  * Compute loss using sparse categorical cross entropy
#  * Report the performance during learning using accuracy
model.compile( optimizer=opt, loss=lossFunction, metrics=['accuracy'])

# Perform the induction
trainingHistory = model.fit(trainX, trainY, epochs=15)


### 4) Evaluate the Model on the Test Data

# Show the testing accuracy over the un-thresholded model:
print("Evaluate DS2")
model.evaluate(testX, testY)