import torch
from prep_dataset.helpers import *
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def test(cnn_model, test_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0
    with torch.no_grad():
        for i,batch in enumerate(test_dataset.dataset):
            """
            for i in batch:
                input, mask, target, path
                input = batch[0][i]
                mask = batch[1][i]
                target = batch[2][i]
                path = batch[3][i]
            """
            #for i, _ in enumerate(batch):
            pixelated_image = torch.from_numpy(batch[0])
            known_array = torch.from_numpy(batch[1])
            target_array = torch.from_numpy(batch[2])
            image_path = batch[3]

            pixelated_image = pixelated_image.to(device)
            target_array = target_array.to(device)

            target_array_normalized = normalize_targets(target_array)
            output = cnn_model(pixelated_image)

            known_array = known_array.to(dtype=torch.bool)
            crop = output[~known_array]
            crop_reshaped = crop.reshape(target_array_normalized.shape)

            loss = criterion(crop_reshaped, target_array_normalized)
            total_loss += loss.item()

            # Comparing the orginal picture with the one provided by the model
            pixelated_image[~known_array] = crop
            pil_image_model = transforms.ToPILImage()(pixelated_image.cpu().detach())
            image_path = Image.open(image_path)
            #pil_image_model.show()
            #image_path.show()

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].imshow(image_path)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(pil_image_model)
            axs[1].set_title("Model Output")
            axs[1].axis("off")

            plt.tight_layout()
            plt.show()

    return total_loss / len(test_dataset)