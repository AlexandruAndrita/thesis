import torch
from prep_dataset.helpers import *
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def validation(cnn_model, validation_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0
    with torch.no_grad():
        for i,batch in enumerate(validation_dataset.dataset):
            """
            for i in batch:
                input, mask, target, path
                input = batch[0][i]
                mask = batch[1][i]
                target = batch[2][i]
                path = batch[3][i]
            """
            # for i, _ in enumerate(batch):
            pixelated_image = torch.from_numpy(batch[0])
            known_array = torch.from_numpy(batch[1])
            target_array = torch.from_numpy(batch[2])
            _ = batch[3]

            pixelated_image = pixelated_image.to(device)
            target_array = target_array.to(device)

            target_array_normalized = normalize_targets(target_array)
            pixelated_image=pixelated_image.reshape(1,pixelated_image.shape[0],pixelated_image.shape[1],pixelated_image.shape[2])
            output = cnn_model(pixelated_image)
            output=output.reshape(1,output.shape[2],output.shape[3])
            output_normalized = normalize_targets(output)

            known_array = known_array.to(dtype=torch.bool)
            crop = output_normalized[~known_array]
            crop_reshaped = crop.reshape(target_array_normalized.shape)

            loss = criterion(crop_reshaped, target_array_normalized)
            total_loss += loss.item()

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
    print(f"Total loss train divided by length: {total_loss / len(validation_dataset)}")
    return total_loss / len(validation_dataset)