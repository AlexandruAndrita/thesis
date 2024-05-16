import os
from werkzeug.utils import secure_filename
import io
import base64
from io import BytesIO

from flask import Flask, render_template, request, redirect, flash
import pickle

from PIL import Image
import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
from torchvision import transforms

from evaluation.model_file import CNNModel
from prep_dataset import to_grayscale
from evaluation.train_individual_model import find_model_output


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
model = CNNModel()
model.load_state_dict(torch.load("CNNModel.pth"))
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', "JPG", "JPEG"]

def allowed_pickle(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ['pkl']

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
            flash('No file part')
            return render_template('index.html')
        file = request.files['file']
        pickle_file = request.files['pickle_file']
        if file.filename == '':
            flash('No image selected')
            return render_template('index.html')
        if pickle_file.filename == '':
            flash('No boolean mask selected')
            return render_template('index.html')
        if file and allowed_file(file.filename) and allowed_pickle(pickle_file.filename):
            filename = secure_filename(file.filename)
            file_content = file.read()
            pickle_content = pickle_file.read()
            encoded_pickle = base64.b64encode(pickle_content).decode('utf-8')
            encoded_img = base64.b64encode(file_content).decode('utf-8')
            return render_template('index.html', filename=encoded_img, pickle_file=encoded_pickle)
        else:
            flash('Invalid file type.')
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    filename = request.form['filename']
    pickle_file = request.form['pickle_file']
    pickle_bytes = base64.b64decode(pickle_file)
    known_array = pickle.loads(pickle_bytes)
    # preprocess = transforms.Compose([
    #     transforms.Resize((128, 170)),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.485], std=[0.229])
    # ])
    image = base64.b64decode(filename)
    img = Image.open(io.BytesIO(image))
    np_arr = np.array(img)
    grayscale_img = to_grayscale.to_grayscale(np_arr)
    grayscale_img = grayscale_img.reshape(grayscale_img.shape[1],grayscale_img.shape[2],1).mean(axis=2)
    known_array = known_array.squeeze(0)
    final_image = find_model_output(
        regressor=KNeighborsRegressor(n_neighbors=10, weights='distance'),
        known_array = known_array,
        image = grayscale_img
    )
    final_image_pil = Image.fromarray(final_image.astype(np.uint8))
    # grayscale_img = grayscale_img.squeeze(0)
    # grayscale_img_pil = Image.fromarray(grayscale_img)
    # img_tensor = preprocess(grayscale_img_pil)
    # with torch.no_grad():
    #     output_tensor = model(img_tensor)
    #
    # output_tensor = output_tensor.squeeze(0)
    # processed_output = output_tensor.cpu().detach().numpy()
    # processed_img_pil = Image.fromarray((processed_output * 255).astype(np.uint8))

    buffered = BytesIO()
    final_image_pil.save(buffered, format="JPEG")
    processed_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template('index.html', filename=filename, processed_filename=processed_img_str)


@app.route('/save_image', methods=['POST'])
def save_image_route():
    # in bytes
    filename = request.form['filename']
    decoded_img = base64.b64decode(filename)
    saved_path = save_image(decoded_img, 'processed_image.jpg')
    flash(f'Processed image saved successfully')
    return redirect('/')


@app.route('/discard_images', methods=['POST'])
def discard_images():
    flash('Images discarded successfully')
    return redirect('/')


@app.route('/discard_input_image', methods=['POST'])
def discard_input_image():
    flash('Input image discarded successfully')
    return redirect('/')


if __name__ == '__main__':
    app.run()


