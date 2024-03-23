import torch
from prep_dataset.helpers import *


def validation(cnn_model, validation_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0
    index = 0
    with torch.no_grad():
        for batch in validation_dataset:
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
                _ = batch[3][i]

                pixelated_image = pixelated_image.to(device)
                target_array = target_array.to(device)

                target_array_normalized = normalize_targets(target_array)
                output = cnn_model(pixelated_image)

                known_array = known_array.to(dtype=torch.bool)
                crop = output[~known_array]
                crop_reshaped = crop.reshape(target_array_normalized.shape)

                loss = criterion(crop_reshaped, target_array_normalized)
                total_loss += loss.item()

                # supplimentary check that should be removed afterwards
                if i==len(batch[3])-1:
                    break

    print(f"Total loss train: {total_loss}")
    print(f"Total loss train divided by length: {total_loss / len(validation_dataset)}")
    return total_loss / len(validation_dataset)