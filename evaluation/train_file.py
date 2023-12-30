import numpy as np
import torch
from prep_dataset.helpers import *

def train(cnn_model, train_dataset, optimizer, criterion, device):
    cnn_model.train()
    #index = 0
    total_loss = 0
    for batch in train_dataset:
        """
        input, _, target, _
        """
        pixelated_image,know_array,target_array,path = batch
        pixelated_image=pixelated_image.to(device)
        target_array=target_array[0].to(device)
        target_array_normalized = normalize_targets(target_array)

        optimizer.zero_grad() # reset accumulated gradients
        output = cnn_model(pixelated_image)

        know_array = know_array.to(dtype=torch.int32)

        output_masked = output[know_array.expand_as(output)].view(-1)
        target_masked = target_array[know_array.expand_as(target_array)].view(-1)

        print(f"Input shape: {pixelated_image.shape}")
        print(f"Boolean mask: {know_array.shape}")
        print(f"Target shape: {target_array_normalized.shape}")
        print(f"Output shape: {output.shape}")

        loss = criterion(output_masked, target_masked)

        loss.backward() # compute gradients
        optimizer.step() # weight update
        total_loss += loss.item()

        #index += 1

    return total_loss/len(train_dataset)

