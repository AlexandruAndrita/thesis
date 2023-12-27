def train(cnn_model, train_dataset, optimizer, criterion, device):
    cnn_model.train()
    total_loss = 0
    for batch in train_dataset:
        """
        input, _, target, _
        """
        pixelated_image,know_array,target_array,path = batch
        pixelated_image=pixelated_image.to(device)
        target_array=target_array.to(device)

        optimizer.zero_grad()
        output = cnn_model(pixelated_image)
        loss = criterion(output, target_array)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(train_dataset)

