import requests
import json
import os
import sys
import numpy as np
from flask import Flask, render_template, request
from image_preprocessing import *
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
MODEL_PATH = './model/1'
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return "go to predict"

@app.route('/predict')
def hello():
    image_url = request.args.get('image_url', default='')
    img = analyze_image(image_url)
    if img is None:
        return "Bad link or path"

    res = model.predict(np.array([img]))
    output = {
        'raw predictions': (str(res[0][0]), str(res[0][1])),
        'label': CLASS_NAMES[np.argmax(res[0])]
    }

    pred = json.dumps(output)
    print(pred)
    return pred

if __name__ == '__main__':
    app.run(host='0.0.0.0')