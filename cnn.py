import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(150, 150, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten());
classifier.add(Dense(120, activation='relu'))
classifier.add(Dense(84, activation='relu'))
classifier.add(Dense(6, activation='softmax'))
# Part 2 - Fitting the CNN to the images
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/saurav/Documents/Saurav/MLDL/practice/cnn/archived_data/intelImage/archive/seg_train/seg_train',
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/home/saurav/Documents/Saurav/MLDL/practice/cnn/archived_data/intelImage/archive/seg_test/seg_test',
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 5,
                         validation_data = test_set,    
                         validation_steps = 500)

classifier.save("lenet2.h5")
print("Saved model to disk")

# Part 3 - Making new predictions




