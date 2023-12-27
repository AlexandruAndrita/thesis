import torch


def validation(cnn_model, validation_dataset, criterion, device):
    cnn_model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in validation_dataset:
            """
            input, _, target, _
            """
            pixelated_image, know_array, target_array, path = batch
            pixelated_image = pixelated_image.to(device)
            target_array = target_array.to(device)

            output = cnn_model(pixelated_image)
            loss = criterion(output, target_array)
            total_loss += loss.item()

    return total_loss / len(validation_dataset)