from flask import Flask, render_template, request, send_file
import torch
from evaluation.model_file import CNNModel
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

def preprocess_image(image):
    # image=image.convert("L")
    # image=image.resize((28,28))
    # image=np.array(image)/255.0
    # image=torch.tensor(image,dtype=torch.float32)
    return image

def predict(image):
    model = CNNModel()
    model.load_state_dict(torch.load("CNNModel.pth"))
    model.eval()
    # feed the image to the model and get the result image
    return image

@app.route("/process_image", methods=["POST","GET"])
def index():
    if request.method == "POST":
        # Get the uploaded image from the request
        uploaded_image = request.files["imageUploadedDisplayed"]

        # Read the image using PIL
        image = Image.open(uploaded_image)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Apply the CNN model
        result_image = predict(preprocessed_image)

        # Convert the result image back to PIL format
        result_image_pil = Image.fromarray(result_image)

        # Save the result image to a byte stream
        result_image_stream = io.BytesIO()
        result_image_pil.save(result_image_stream)
        result_image_stream.seek(0)

        # Return the result image as a response
        return send_file(result_image_stream)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
