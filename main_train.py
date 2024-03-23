from group_dataset_images.prepare_datasets import get_images
from evaluation.model_file import CNNModel
from evaluation.train_file import train
from evaluation.validation_file import validation
from evaluation.test_file import test
import torch
import torch.nn as nn


if __name__ == '__main__':
    input_directory_path = "D:\\an III\\bachelor's thesis\\thesis\\dataset\\test"
    train_dataset,validation_dataset,test_dataset = get_images(input_directory_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn_model = CNNModel()
    cnn_model.to(dtype=torch.float64,device=device)
    optimizer = torch.optim.Adam(cnn_model.parameters())
    criterion = nn.MSELoss() # loss function

    num_epochs = 5
    for epoch in range(num_epochs):
        loss_train = train(cnn_model, train_dataset, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss_train}")
        loss_validation = validation(cnn_model, validation_dataset, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss_train} - Validation Loss: {loss_validation}")

    loss_test = test(cnn_model, test_dataset, criterion, device)
    print(f"Test Loss: {loss_test}")

    torch.save(cnn_model.state_dict(),"CNNModel.pth")
    print("Model saved successfully under the name: CNNModel.pth")
