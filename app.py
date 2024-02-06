from flask import Flask, render_template, request
import torch
from evaluation.model_file import CNNModel

app = Flask(__name__)

def predict(image):
    """
    i. preprocessing of the image values before calling the ML model
    ii. load the trained model
    iii. preprocess image
    iv. fed the processed image in the trained model
    v. convert the image back to human-readable format with PIL
    vi. return the image after
    """
    model = CNNModel()
    model.load_state_dict(torch.load("CNNModel.pth"))
    model.eval()
    # write job in order to feed the image back to the model

    return image

@app.route("/", methods=["POST","GET"])
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
