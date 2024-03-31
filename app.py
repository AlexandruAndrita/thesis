import os
from flask import Flask, render_template, request, redirect
from evaluation.model_file import CNNModel
from PIL import Image
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', "JPG", "JPEG"]


def grayscale_image(image):
    img = Image.open(io.BytesIO(image)).convert('L')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_img


def save_image(image, filename):
    downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    image_path = os.path.join(downloads_dir, filename)
    with open(image_path, 'wb') as f:
        f.write(image)
    return image_path


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            # flash('No file part')
            return render_template('index.html')
        file = request.files['file']
        if file.filename == '':
            # flash('No selected file')
            return render_template('index.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_content = file.read()
            # flash('File successfully uploaded')
            encoded_img = base64.b64encode(file_content).decode('utf-8')
            return render_template('index.html', filename=encoded_img)
        # else:
            # flash('Invalid file type. Please upload a JPG or JPEG image.')
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    filename = request.form['filename']
    decoded_img = base64.b64decode(filename)
    processed_img = grayscale_image(decoded_img)
    return render_template('index.html', filename=filename, processed_filename=processed_img)


@app.route('/save_image', methods=['POST'])
def save_image_route():
    # in bytes
    filename = request.form['filename']
    decoded_img = base64.b64decode(filename)
    saved_path = save_image(decoded_img, 'processed_image.jpg')
    # flash(f'Processed image saved successfully at: {saved_path}')
    return redirect('/')


@app.route('/discard_images', methods=['POST'])
def discard_images():
    # flash('Images discarded successfully')
    return redirect('/')


@app.route('/discard_input_image', methods=['POST'])
def discard_input_image():
    # flash('Input image discarded successfully')
    return redirect('/')


if __name__ == '__main__':
    app.run()


