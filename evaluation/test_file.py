import torch
from prep_dataset.helpers import *
from torchvision import transforms
from PIL import Image


def test(cnn_model, test_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_dataset:
            """
            for i in batch:
                input, mask, target, path
                input = batch[0][i]
                mask = batch[1][i]
                target = batch[2][i]
                path = batch[3][i]
            """
            for i, _ in enumerate(batch):
                pixelated_image = batch[0][i]
                known_array = batch[1][i]
                target_array = batch[2][i]
                image_path = batch[3][i]

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
                # pixelated_image[~known_array] = crop
                # pil_image_model = transforms.ToPILImage()(pixelated_image.cpu().detach())
                # image_path = Image.open(image_path)
                # pil_image_model.show()
                # image_path.show()

                if i==len(batch[3])-1:
                    break

    return total_loss / len(test_dataset)