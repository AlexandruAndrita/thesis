import torch
from prep_dataset.helpers import *


def validation(cnn_model, validation_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0
    index = 0
    with torch.no_grad():
        for batch in validation_dataset:
            """
            input, _, target, _
            """
            pixelated_image, know_array, target_array, path = batch
            pixelated_image = pixelated_image.to(device)
            target_array=target_array[0].to(device)
            target_array_normalized = normalize_targets(target_array)

            output = cnn_model(pixelated_image)

            loss = criterion(output, pixelated_image)
            total_loss += loss.item()

            index += 1

    return total_loss / len(validation_dataset)