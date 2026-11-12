import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = self.drop2D(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



if __name__ == "__main__":

    training_data = datasets.FashionMNIST(root="C:/TransUnrecon/data", train=True, download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))
    validation_data = datasets.FashionMNIST(root="C:/TransUnrecon/data", train=False, download=True,
                                       transform=transforms.Compose([transforms.ToTensor()]))
    batch_size = 128
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True)


    criterion = nn.CrossEntropyLoss()
    classifier = CNN().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    num_epochs=10

    for epoch in range(num_epochs):   # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'[{epoch} / {num_epochs}] loss: {running_loss / i:.3f}')
        running_loss = 0.0

        torch.save(classifier.state_dict(), 'C:/TransUnrecon/VAE_MNIST_2/classifier_Fashionmnist.pth')
    print('Finished Training') # 최종 loss 0.044