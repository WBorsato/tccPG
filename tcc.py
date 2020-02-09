import pandas as pd
import numpy as np
import tensorflow as tf
#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

#model.compile(loss = "...", optimizer = "...", metrics = "..", options = run_opts)
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.applications import ResNet50
from tensorflow.keras.models import model_from_json

base_model= ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))

model= Sequential()
model.add(base_model)
model.add(Conv2D(32, (3, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.40))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.40))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator
#train_generator = ImageDataGenerator(rescale = 1./255,
#                                     rotation_range=10,  
#                                     zoom_range = 0.1, 
#                                     width_shift_range=0.1,  height_shift_range=0.1) 
test_generator = ImageDataGenerator(rescale = 1./255)

#training_set = train_generator.flow_from_directory('C:/Users/Welington/Documents/tcc/input/data/train',
#                                                 target_size = (224,224),
#                                                 batch_size = 16,
#                                                 class_mode = 'categorical')

#test_set = test_generator.flow_from_directory('C:/Users/Welington/Documents/tcc/input/data/test',
#                                           target_size = (224, 224),
#                                           batch_size = 16,
#                                           class_mode = 'categorical',
#                                           shuffle=False)

#model.compile(optimizer=optimizers.adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
#
#from keras.callbacks import ReduceLROnPlateau
#learn_control = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=.5, min_lr=0.0001)
#
#model.fit_generator(generator=training_set,
#                            steps_per_epoch=training_set.samples//training_set.batch_size,
#                            validation_data=test_set,
#                            verbose=1,
#                            validation_steps=test_set.samples//test_set.batch_size,
#                            epochs=27,callbacks=[learn_control])
#
#test_set.reset()
#predictions = model.predict_generator(test_set, steps=test_set.samples/test_set.batch_size,verbose=1)
#y_pred= np.argmax(predictions, axis=1)
#
#print(y_pred)
#
#y_test=np.array([])
#for i in range(360):
#    y_test=np.append(y_test,0)
#for i in range(300):
#    y_test=np.append(y_test,1)
#
#    from sklearn.metrics import confusion_matrix 
#cm= confusion_matrix(y_test,y_pred)
#
#print(cm)
#
## serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def formatImage(file):
   np_image = file
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

#image = load('C:/Users/Welington/Documents/tcc/input/data/singletest/1.jpg')
#print(image)
#y_pred= loaded_model.predict(image)
#print(y_pred)




#predictions = loaded_model.predict_generator(test_set, steps=test_set.samples/test_set.batch_size,verbose=1)
#y_test=np.array([])
#for i in range(360):
#    y_test=np.append(y_test,0)
#for i in range(300):
#    y_test=np.append(y_#test,1)

#    from sklearn.metrics import confusion_matrix 
#cm= confusion_matrix(y_test,#y_pred)

#print(cm)




#API

import cv2
from flask_restful import Resource, Api
from json import dumps

import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename



ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['POST'])
def upload_file():
 # check if the post request has the file part
 if 'file' not in request.files:
  resp = jsonify({'message' : 'No file part in the request'})
  resp.status_code = 400
  return resp
 file = request.files['file']
 if file.filename == '':
  resp = jsonify({'message' : 'No file selected for uploading'})
  resp.status_code = 400
  return resp
 if file and allowed_file(file.filename):
  #print(app.config['UPLOAD_FOLDER'])
  #print(file.filename)
  filename = secure_filename(file.filename) 
  file.save(os.path.join(os.path.abspath(os.curdir), filename))  
  print(os.path.join(os.path.abspath(os.curdir)+'/'+filename))
  path = os.path.join(os.path.abspath(os.curdir)+'/'+filename)
  print('path = ' + path)

  image = load(path)
  print(image)
  y_pred= loaded_model.predict(image)
  print(y_pred)

  #image = load(path)
  #print(image)
  #y_pred= loaded_model.predict(image)
  #print(y_pred)
  resp = jsonify({'message' : 'File successfully uploaded','Benigno':str(y_pred[0][0]),'Maligno':str(y_pred[0][1])})
  resp.status_code = 201
  return resp
 else:
  resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
  resp.status_code = 400
  return resp

if __name__ == "__main__":
    app.run(port=7000)