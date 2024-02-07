import os
from flask import Flask, render_template, request, send_file
import torch
from evaluation.model_file import CNNModel
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

def preprocess_image(image):
    image=image.convert("L")
    return image

def predict(image):
    # load image, feed the image to the model, get the result image

    #model = CNNModel()
    #model.load_state_dict(torch.load("CNNModel.pth"))
    #model.eval()

    return image

@app.route("/", methods=["POST","GET"])
def index():
    if request.method == "POST":
        try:
            uploaded_image = request.files["imageUploadedDisplayed"]

            image = Image.open(uploaded_image)
            image_format = image.format
            preprocessed_image = preprocess_image(image)
            result_image = predict(preprocessed_image)
            result_image_stream = io.BytesIO()
            result_image.save(result_image_stream,format=image_format)
            result_image_stream.seek(0)

            mimetype_value = "image/"+image_format.lower()

        except Exception as e:
            print(f"Exception caught: {e}")

        return send_file(result_image_stream,mimetype=mimetype_value)

    elif request.method == "GET":
        pass
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
