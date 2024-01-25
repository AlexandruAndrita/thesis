from flask import Flask, render_template, request


app = Flask(__name__)

def predict(image):
    """
    i. preprocessing of the image values before calling the ML model
    ii. load the trained model
    iii. fed the image in the trained model
    iv. convert the image back to human-readable format with PIL
    v. return the image after
    """
    return image

@app.route("/", methods=["POST","GET"])
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
