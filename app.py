from flask import Flask, request, jsonify, render_template,flash
import os
import tensorflow as tf
import pickle
import numpy as np
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "super secret key"
global samples_model

# Define the path to the saved model
samples_model = load_model('C://Users//RUTUJA//MSc sem 3 project//trainedmodel.h5')

#saved_model_path = pickle.load(open('C://Users//RUTUJA//MSc sem 3 project//model//trained_model_samples.pkl','rb'))
#encoded_path = saved_model_path.encode('utf-8')

#file_path = 'C:/Users/RUTUJA/MSc sem 3 project/model/trained_model_samples.pkl'

#with open(file_path , 'rb') as f:
#    dict1 = pickle.load(f)
    
# Define the io_device
io_device = "/job:localhost"

# Create a LoadOptions object with the io_device option set
load_options = tf.saved_model.LoadOptions(experimental_io_device=io_device)

# Load the saved model using the specified options
loaded_model = tf.keras.models.load_model(samples_model, options=load_options)

# Load the trained OCR model
#samples_model = pickle.load(open('C:/Users/RUTUJA/MSc sem 3 project/trained_model_samples.pkl', 'rb'))  # Replace with your model file

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST','GET'])
class LayerCTC(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost


    def call(self, y_true, y_pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")


        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")


        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)


        # Return Computed predictions
        return y_pred

def process_image():
        
# Get the uploaded image file
    file = request.files['file']
    img_path = os.path.join('C:/Users/RUTUJA/MSc sem 3 project/samples/',file.filename)
    img = tf.io.read_file(img_path)
    # Converting the image to grayscale
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resizing to the desired size
    img = tf.image.resize(img, [200, 75])
    # Transposing the image
    img = tf.transpose(img, perm=[1, 0, 2])
    samples_model = tf.keras.models.load_model('C:/Users/RUTUJA/MSc sem 3 project/model/captcha.h5')#, custom_objects={'LayerCTC': CTCLayer})
    # Mapping image label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    
    return {"image": img, "label": label}


    if file.filename == '':
        flash('no image')
    else:
        flash(file.filename)
    return render_template(img)

        

if __name__ == '__main__':
    app.run(debug=True)
