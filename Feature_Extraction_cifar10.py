# Import the libraries
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, UpSampling2D,Conv2D, MaxPooling2D,Input, Dense,BatchNormalization,Input
from keras.datasets import cifar10 
from keras.applications import DenseNet201,ResNet50,VGG19,DenseNet121
from keras.optimizers import SGD,Adam,RMSprop
from keras_efficientnets import EfficientNetB5,EfficientNetB7
import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer

# Load the dataset
(x_train, Y_train), (x_test, Y_test) = cifar10.load_data()

# convert labels into one-hot encoding
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test= np_utils.to_categorical(Y_test, 10)

# data-normalization in range 0 to 1
x_train = x_train/255.
x_test = x_test/255.
print(x_train.shape,Y_test.shape)

################### Feature extract from DenseNet121 #################
base = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
for layer in base.layers:
    base.Trainable = False
x = Flatten()(base.output)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu',name='flattenn')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
prediction = Dense(10, activation='softmax')(x)

##### build the model and run it ###########
model = Model(inputs=base.input, outputs=prediction)
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr = 0.001,momentum = 0.9),metrics=['acc'])
model.fit(x_train, Y_train, epochs=40,shuffle=True, batch_size=256, verbose=0,validation_data=(x_test, Y_test))

##### generate new model from the existing model as a ouput hidden layer to extact 64 features
new_model = Model(model.inputs, model.get_layer('flattenn').output)
# extract training and testing features 
train_features = new_model.predict(x_train)
test_features = new_model.predict(x_test)
print(train_features.shape)

# scaling the training/teting data into range of 0 to 1.

scaler = MinMaxScaler(feature_range=(0,1))
normalize_train_features = scaler.fit_transform(train_features)
normalize_test_features = scaler.transform(test_features)
print(normalize_train_features.shape)
print(normalize_test_features.shape)

# transpose for cuda ELM
train = normalize_train_features.transpose()
test = normalize_test_features.transpose()
Y_train_T = Y_train.transpose()
Y_test_T = Y_test.transpose()

### save the data in csv file for ELM-CUDA
np.savetxt('cuda_elm/features_cifar10/train_features.csv', train, delimiter=',')
np.savetxt('cuda_elm/features_cifar10/test_features.csv', test, delimiter=',')
np.savetxt('cuda_elm/features_cifar10/train_labels.csv', Y_train_T, delimiter=',')
np.savetxt('cuda_elm/features_cifar10/test_labels.csv',Y_test_T, delimiter=',')

############ try with ELM on CPU and compare with our ELM-CUDA #################
# convert back to original labels
y_train = np.argmax(Y_train, axis=-1)
y_test = np.argmax(Y_test, axis=-1)
clf = GenELMClassifier(hidden_layer=MLPRandomLayer(n_hidden=1000, activation_func='tanh')) 
clf.fit(normalize_train_features, y_train)
res = clf.score(normalize_test_features, y_test)
print("ELM score:",res*100)