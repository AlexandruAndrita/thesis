import math

from group_dataset_images.prepare_datasets import get_images
from evaluation.model_file import CNNModel
from evaluation.train_file import train
from evaluation.validation_file import validation
from evaluation.test_file import test
import torch
import torch.nn as nn
import glob
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from prep_dataset import helpers
from group_dataset_images import group_images
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # option = int(input("1 = train\n2 = test\nYour option: "))

    option = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn_model = CNNModel()
    try:
        cnn_model.load_state_dict(torch.load("CNNModel.pth"))
    except Exception as e:
        pass
    cnn_model.to(dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(cnn_model.parameters(),lr=0.01,weight_decay=1e-5)
    criterion = nn.MSELoss()

    if option == 1:
        input_directory_path = "D:\\an III\\bachelor's thesis\\thesis\\dataset\\test_v22"
        #input_directory_path = "D:\\an III\\bachelor's thesis\\resized_images"
        #input_directory_path = "D:\\an III\\bachelor's thesis\\thesis\\dataset\\test"
        #input_directory_path = "C:\\Users\\A.Andrita\\Downloads\\test"
        #input_directory_path = "C:\\Users\\A.Andrita\\Downloads\\test"
        train_dataset,validation_dataset,test_dataset = get_images(input_directory_path)

        print(f"Train dataset size: {len(train_dataset.sampler)}")
        print(f"Validation dataset size: {len(validation_dataset.sampler)}")
        print(f"Test dataset size: {len(test_dataset.sampler)}")

        num_epochs = 10
        losses_train = []
        losses_validation = []
        for epoch in range(num_epochs):
            loss_train = train(cnn_model, train_dataset, optimizer, criterion, device)
            loss_validation = validation(cnn_model, validation_dataset, criterion, device)

            if math.isnan(loss_train) or math.isnan(loss_validation):
                print(f"Stopping training at epoch {epoch}/{num_epochs} because of NaN values.")
                print(f"Train loss: {loss_train}")
                print(f"Validation loss: {loss_validation}")
                num_epochs = epoch + 1
                break

            losses_train.append(loss_train)
            losses_validation.append(loss_validation)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss_train} - Validation Loss: {loss_validation}")

        if num_epochs != 0 and num_epochs != 1:
            plt.plot(range(1,num_epochs+1),losses_train,label="Train Loss")
            plt.plot(range(1,num_epochs+1),losses_validation, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Plot with training losses vs. validation losses was not done since loss values are NaN")

        loss_test = test(cnn_model, test_dataset, criterion, device)
        print(f"Test Loss: {loss_test}")

        torch.save(cnn_model.state_dict(),"CNNModel.pth")
        print("Model saved successfully under the name: CNNModel.pth")

    elif option == 2:
        input_directory_path = "D:\\an III\\bachelor's thesis\\thesis\\dataset\\testing_v1"

        preprocessing = transforms.Compose([
            transforms.ToTensor()
        ])

        images = group_images.RandomImagePixelationDataset(input_directory_path,
                                                           width_range=(4, 32),
                                                           height_range=(4, 32),
                                                           size_range=(4, 16))

        fig,axs = plt.subplots(len(images), 2, figsize=(10,5))

        for i,image in enumerate(images):
            pix_image=image[0]
            target_array = image[2]
            #tmp = pix_image.astype(dtype=np.uint8)
            tmp = pix_image.reshape(pix_image.shape[1],pix_image.shape[2])
            input_pil = Image.fromarray(tmp)

            input_tensor = preprocessing(pix_image).to(device)
            #input_ = helpers.normalize_targets(input_tensor)
            input_ = torch.clone(input_tensor).to(device)
            input_ = input_.reshape(1,pix_image.shape[1],pix_image.shape[2])
            input_ = input_.to(dtype=torch.float64)

            # input_ = torch.clone(input_tensor).to(device)
            # input_ = input_.reshape(1, pix_image.shape[1], pix_image.shape[2])
            # input_ = input_.to(dtype=torch.float64)

            with torch.no_grad():
                output = cnn_model(input_)

            mini_pix = target_array.min()
            maxi_pix = target_array.max()

            output_tmp = output*(maxi_pix-mini_pix)+mini_pix
            output_tmp = output_tmp.cpu().detach().numpy()
            output_tmp = output_tmp.reshape(pix_image.shape[1],pix_image.shape[2])

            output_pil = Image.fromarray(output_tmp)

            axs[i,0].imshow(input_pil)
            axs[i,0].set_title("Original Image")
            axs[i,0].axis("off")

            axs[i,1].imshow(output_pil)
            axs[i,1].set_title("Model Output")
            axs[i,1].axis("off")

        plt.tight_layout()
        plt.show()

