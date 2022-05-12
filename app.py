import base64
import numpy as np
import io
from PIL import Image
# for our model
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
# to retrieve and send back data
from flask import request
from flask import jsonify
from flask import Flask

# create a variable named app
app = Flask(__name__) 

# create our model
IMG_SHAPE = (224, 224, 3)
def get_model():
    model = ResNet50(include_top=True, weights="imagenet", input_shape=IMG_SHAPE)
    print("[+] model loaded")
    return model

# decode the imaeg coming from the request
def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    return decoded

# preprocess image before sending it to the model
def preprocess(decoded):
    #resize and convert in RGB in case image is in RGBA
    pil_image = Image.open(io.BytesIO(decoded)).resize((224,224), Image.LANCZOS).convert("RGB") 
    image = np.asarray(pil_image)
    batch = np.expand_dims(image, axis=0)
    
    return batch

# load model so it's in memory and not loaded each time there is a request
model = get_model()
  
# function predict is called at each request  
@app.route("/predict", methods=["POST"])
def predict():

    print("[+] request received")

    # get the data from the request and put ir under the right format

    req = request.get_json(force=True)

    image = decode_request(req)

    batch = preprocess(image)

    # actual prediction of the model

    prediction = model.predict(batch)

    # get the label of the predicted class

    print(decode_predictions(prediction))

    a=decode_predictions(prediction)

    b=a[0][0]

    top_label =[(b[1],str(b[2]))]

    #top_label = [(i[1],str(i[2])) for i in decode_predictions(prediction)[0][0]]

    print(top_label)

    # create the response as a dict

    response = {"prediction": top_label}

    print("[+] results {}".format(response))

    

    return jsonify(response) # return it as json