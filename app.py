import numpy as np
from PIL import Image
from flask import Flask, render_template, request

from camera import image_predict

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def index():
    pred = None
    message = None
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.convert('L')
                img = img.resize((50,50))
                img = np.asarray(img)
                print(img.shape)
                img = img.reshape((1,50,50,1))
                img = img/255.0
                pred = image_predict(img)
        except:
            message = "Upload an Image for Prediction"
            return render_template('index.html', message = message)
    return render_template("index.html", pred = pred, message = message)

if __name__=='__main__':
    app.run(debug=True)
