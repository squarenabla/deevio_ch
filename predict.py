import requests
import json
from flask import Flask, render_template, request
from image_preprocessing import *

app = Flask(__name__)

@app.route('/predict')
def hello():
    #name=None ensures the code runs even when no name is provided
    image_url = request.args.get('image_url', default='')

    img = analyze_image(image_url)
    if img is None:
        return "Bad link or path"
    # Making POST request
    payload = {"instances": [img.tolist()]}

    r = requests.post('http://localhost:9000/v1/models/Resnet:predict', json=payload)
    pred = json.loads(r.content.decode('utf-8'))
    pred['labels'] = CLASS_NAMES[np.argmax(pred['predictions'][0])]
    print(pred)
    return pred
