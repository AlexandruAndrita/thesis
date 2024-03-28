import numpy as np
import torch
from prep_dataset.helpers import *
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def train(cnn_model, train_dataset, optimizer, criterion, device):
    cnn_model.train()
    total_loss = 0
    for i,batch in enumerate(train_dataset.dataset):
        """
        for i in batch:
            input, mask, target, path
            input = batch[0][i]
            mask = batch[1][i]
            target = batch[2][i]
            path = batch[3][i]
        """
        #for i,_ in enumerate(batch):
        pixelated_image = torch.from_numpy(batch[0])
        known_array = torch.from_numpy(batch[1])
        target_array = torch.from_numpy(batch[2])
        _ = batch[3]

        pixelated_image = pixelated_image.to(device)
        target_array=target_array.to(device)

        target_array_normalized = normalize_targets(target_array)
        optimizer.zero_grad() # resetting accumulated gradients
        pixelated_image=pixelated_image.reshape(1,pixelated_image.shape[0],pixelated_image.shape[1],pixelated_image.shape[2])
        output = cnn_model(pixelated_image)
        output=output.reshape(1,output.shape[2],output.shape[3])
        output_normalized = normalize_targets(output)

        # print(f"Input shape: {pixelated_image.shape}")
        # print(f"Boolean mask: {known_array.shape}")
        # print(f"Target shape: {target_array_normalized.shape}")
        # print(f"Output shape: {output.shape}")

        known_array = known_array.to(dtype=torch.bool)
        crop = output_normalized[~known_array]
        crop_reshaped = crop.reshape(target_array_normalized.shape)

        loss = criterion(crop_reshaped, target_array_normalized)
        total_loss += loss.item()

        loss.backward()  # compute gradients
        optimizer.step()  # weight update

        # # Comparing the orginal picture with the one provided by the model
        # pixelated_image_normalized = normalize_targets(pixelated_image)
        # pixelated_image_normalized[~known_array] = crop
        # pil_image_model = transforms.ToPILImage()(pixelated_image_normalized.cpu().detach())
        # image_path = Image.open(image_path)
        # #pil_image_model.show()
        # #image_path.show()

        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # axs[0].imshow(image_path)
        # axs[0].set_title("Original Image")
        # axs[0].axis("off")

        # axs[1].imshow(pil_image_model,cmap="gray")
        # axs[1].set_title("Model Output")
        # axs[1].axis("off")

        # plt.tight_layout()
        # plt.show()
        

    print(f"Total loss train: {total_loss}")
    print(f"Total loss train divided by length: {total_loss / len(train_dataset)}")

    return total_loss / len(train_dataset)

def replace_pixelated_area(pixelated_image,known_array,target_array):

    start_row,start_col=99999,99999
    end_row, end_col=0,0
    N=pixelated_image.shape[0]

    tmp_known_array=np.array(known_array.tolist())
    tmp_known_array=np.max(tmp_known_array,axis=0).squeeze()

    #tmp_known_array=tmp_known_array.reshape(pixelated_image.shape[2],pixelated_image.shape[3])

    tmp=np.array(pixelated_image.tolist())
    tmp=np.max(tmp,axis=0).squeeze()

    #tmp=tmp.reshape(pixelated_image.shape[2],pixelated_image.shape[3])

    for row in range(pixelated_image.shape[2]):
        for col in range(pixelated_image.shape[3]):
            if not tmp_known_array[row][col]:
                start_row=min(start_row,row)
                start_col=min(start_col,col)
                end_row=max(end_row,row)
                end_col=max(end_col,col)

    if start_col!=0 and start_row!=0 and end_col!=0 and end_row!=0:
        tmp[start_row:end_row+1,start_col:end_col+1]=np.array(target_array[0].tolist())

    tmp=np.expand_dims(tmp,axis=0)
    tmp=np.repeat(tmp,N,axis=0)
    tmp=torch.tensor(tmp)

    return tmp


