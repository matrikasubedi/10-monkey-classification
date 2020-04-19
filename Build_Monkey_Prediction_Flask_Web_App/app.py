#Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
from tensorflow.python.keras import backend as k

# Create flask instance
app = Flask(__name__)

class_labels = [
	'mantled_howler', 
	'patas_monkey', 
	'bald_uakari', 
	'japanese_macaque', 
	'pygmy_marmoset', 
	'white_headed_capuchin',
	'silvery_marmoset',
	'common_squirrel_monkey', 
	'black_headed_night_monkey',
	'nilgiri_langur' 
	]

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.Graph()

# Function to load and prepare the image in right shape
def read_image(filename):
    # Load the image
    img = load_img(filename, target_size=(224, 224))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image into a sample of 1 channel
    img = np.expand_dims(img, axis=0)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images', filename)
                file.save(file_path)
                img = read_image(file_path)
                # Predict the class of an image

                with graph.as_default():
                    model1 = load_model('monkey_breed_mobilNet.h5')
                    class_prediction = np.argmax(model1.predict(img))
                    print(class_prediction)

                #Map apparel category with the numerical class
                product = (class_labels[class_prediction])
                return render_template('predict.html', product = product, user_image = file_path)
        
    return render_template('predict.html')

if __name__ == "__main__":
    init()
    app.run()