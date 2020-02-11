import numpy as np
from tensorflow.keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
#print("Loaded model from disk")

from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image
import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['GET'])
def recebeu_get():
   return jsonify({'message' : 'recebido'})


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
  #filename = secure_filename(file.filename) 
  #file.save(os.path.join(os.path.abspath(os.curdir), filename))  
  #path = os.path.join(os.path.abspath(os.curdir)+'/'+filename)
  image = load(file)
  #print('file')
  #img = load(file)
  #print(img)
  
  y_pred= loaded_model.predict(image)
  
  resp = jsonify({'message' : 'File successfully uploaded','Benigno':str(y_pred[0][0]),'Maligno':str(y_pred[0][1])})
  resp.status_code = 201
  return resp
 else:
  resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
  resp.status_code = 400
  return resp

if __name__ == "__main__":
    app.run(debug = None,port=os.environ.get('PORT'))