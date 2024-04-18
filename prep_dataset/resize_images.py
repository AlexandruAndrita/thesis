"""
Script used to resize and rename many images from certain directory with the purpose of
further using them while training the Machine Learning model.
"""


from PIL import Image
import os

input_directory = "C:\\Users\\andua\\Desktop\\images"
output_directory = "D:\\an III\\bachelor's thesis\\resized_images"

os.makedirs(output_directory, exist_ok=True)

# (width,height)
target_size = (170, 128)

print("Image resizing started")

for index,filename in enumerate(os.listdir(input_directory)):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG") or filename.endswith(".JPEG"):
        image_path = os.path.join(input_directory, filename)
        img = Image.open(image_path)

        img = img.resize(target_size, Image.Resampling.LANCZOS)
        name = f"image{index}."+filename.split(".")[1]
        print("Image named " + filename + "renamed into " + name)
        output_path = os.path.join(output_directory, name)
        img.save(output_path)

print("All images resized successfully")