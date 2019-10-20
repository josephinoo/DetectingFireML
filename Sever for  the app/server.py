
"""Filename: server.py
"""
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import keras

from keras.models import load_model
from keras.utils import CustomObjectScope

from keras.initializers import glorot_uniform
import tensorflow as tf
graph = tf.get_default_graph()



longitud, altura = 150, 150
modelo = './modelo.h5'
pesos_modelo = './pesos.h5'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)
def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    with graph.as_default():
        array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        json = '{"pred": "Fires"}'
        print("pred: Fires")
    elif answer == 1:
        json = '{"pred": "Normal"}'
        print("pred: Normal")
    elif answer == 2:
        json = '{"pred": "Smoke"}'
        print("pred: Smoke")
    return json
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = predict("./uploads/" + filename)
            return str(result)

            
            #return redirect(url_for('uploaded_file',
            #                        filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)