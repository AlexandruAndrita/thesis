import torch
from prep_dataset.helpers import *


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
            output = cnn_model(pixelated_image)

            known_array = known_array.to(dtype=torch.bool)
            crop = output[~known_array]
            crop_reshaped = crop.reshape(target_array_normalized.shape)

            loss = criterion(crop_reshaped, target_array_normalized)
            total_loss += loss.item()

    print(f"Total loss train: {total_loss}")
    print(f"Total loss train divided by length: {total_loss / len(validation_dataset)}")
    return total_loss / len(validation_dataset)