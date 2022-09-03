from pyexpat import model
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


train_datasets = pd.read_csv("../data/MNIST_kaggle/train.csv")
test_datasets = pd.read_csv("../data/MNIST_kaggle/test.csv")



class MNIST_Dataloader(Dataset):

    def __init__(self, path: string, new_shape: tuple = None, dtype=torch.float32) -> None:
        super().__init__()

        self.dataset = pd.read_csv(path)

        self.n_samples = self.dataset.shape[0]

        # self.X = torch.from_numpy(self.dataset.drop(["label"], axis=1).to_numpy() / 255.0).type(dtype)
        self.X = self.dataset.drop(["label"], axis=1).to_numpy() / 255.0
        self.y = torch.from_numpy(self.dataset["label"].to_numpy())
        

        if new_shape != None:
            self.X = self.X.reshape(new_shape)
        self.X = torch.from_numpy(self.X).type(dtype).unsqueeze(1)
        print(self.X.shape)

            
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


train_data = MNIST_Dataloader(
    "../data/MNIST_kaggle/train.csv", new_shape=(42000, 28, 28))
test_data = MNIST_Dataloader("../data/MNIST_kaggle/train.csv", new_shape=(42000, 28, 28))





train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)






class VGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()

        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer_6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
       
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2304, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 10))

    def forward(self, x):
        out = self.layer_1(x.to(device))
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


device = "cuda" if torch.cuda.is_available() else "cpu"


# Optimize model parameters
class Main_MIST_classifier:
    def __init__(self, model):
        self.model = model().to(device)
        print(self.model)
        self.loss_fnn = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0009, weight_decay=0.1)

    def forward(self, dataloader):

        # get the size of dataset in dataloader object
        size = len(dataloader.dataset)

        # switch model to training mode
        self.model.train()

        for batch, (X, y) in enumerate(dataloader):

            # calculate prediction (forward pass)
            prediction = self.model(X)

            # evaluate loss function
            loss = self.loss_fnn(prediction.to(device), y.to(device))

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def evaluate(self, dataloader):
        # get the size of dataset in dataloader object
        size = len(dataloader.dataset)
        print(size)

        num_batches = len(dataloader)

        # switch model to evaluation mode
        self.model.eval()

        test_loss, correct = 0, 0

        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)

                test_loss += self.loss_fnn(pred, y).item()

                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(
                f"Test Error: \n Accuracy: {(100*correct ):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train(self, epochs=10):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.forward(train_loader)
            self.evaluate(test_loader)

            torch.save(self.model.state_dict(), "./CNN_pytorch")
        print("Done!")


model = Main_MIST_classifier(VGG16).train()
