from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from datetime import datetime
import glob
import os
import random

from PIL import Image
import cv2
import numpy as np

import torch

from predict import identify

app = Flask(__name__)

def clear(dir_path):
    for f in glob.glob(os.path.join(dir_path, '*')):
        os.remove(f)

def load_model():
    global classifier
    print(" * Loading pre-trained model ...")
    device = torch.device('cpu')
    classifier = torch.load('model/model.pth', map_location=device)
    print('* Loading End')

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/dialect', methods=['POST'])
def dialect():
    return render_template('./dialect.html')

@app.route('/result', methods=['POST'])
def result():
    if request.files['audio']:
        save_img_url = os.path.join('./static/upload', str(random.randint(0, 10000000)).zfill(10)+'.png')
        _, _, dialect = identify(classifier, request.files['audio'], save_img_url)  
        return render_template('./result.html', 
                               title='エセ関西弁判定器', 
                               dialect=dialect, 
                               save_img_url=save_img_url)

if __name__ == '__main__':
    load_model()
    app.debug = True
    app.run(host='localhost', port=5000)
    clear('./static/upload')