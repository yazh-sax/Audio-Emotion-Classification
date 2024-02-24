import torch.nn as nn
import torch.nn.functional as f
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

TEST_NUM = 0
EPOCHS = 20
LEARNING_RATE = 1.0
OPTIM_GAMMA = 0.8
BATCH_SIZE = 1


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 2, 1)
        self.conv2 = nn.Conv2d(4, 8, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 1, 1)
        self.fc1 = nn.Linear(288, 72)
        self.fc2 = nn.Linear(72, 3)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = f.log_softmax(x, dim=1).to(torch.float32)
        return x


class CSVDataset(Dataset):
    def __init__(self, file_name, label_index):
        # Load X/y Data from .csv file
        file_out = pd.read_csv(file_name)
        y = file_out[label_index]
        x = file_out.iloc[:, 3:]

        # Implement scaling?
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x_train = torch.tensor(self.X.iloc[item].values, dtype=torch.float32).view(4, 7)
        y_train = torch.tensor(self.y.iloc[item], dtype=torch.long)
        return x_train, y_train


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train(model, train_loader, optimizer, device='cpu'):
    model.train()
    running_loss = 0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        running_loss += loss
        count += 1
        loss.backward()
        optimizer.step()
    return running_loss / count


def save_model(_model, _test_num, _epochs_arr, _loss_arr, _acc_arr):
    torch.save(_model, f'3labelmodel{_test_num}.pt')
    plt.scatter(_epochs_arr, loss_arr)
    plt.savefig(f'v{_test_num}_loss')
    plt.show()
    plt.scatter(_epochs_arr, _acc_arr)
    plt.savefig(f'v{_test_num}_accuracy')
    plt.show()


def test(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nEpoch {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Matplotlib
    return 100. * correct / len(test_loader.dataset)


def final_test(model, test_loader, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(f"Prediction: {pred}, Actual: {target}")
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Matplotlib
    return 100. * correct / len(test_loader.dataset)


dataset = CSVDataset("new_modified_actorEmotionsTrainCSV.csv", "Emotion (Label)")
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = CNN()

print(f'Model has: {count_parameters(network)} parameters')

optimizer = optim.Adadelta(network.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=1, gamma=OPTIM_GAMMA)

# Matplotlib Arrays
epochs_arr = list(range(1, EPOCHS + 1))
loss_arr = []
acc_arr = []

for i in range(1, EPOCHS + 1):
    loss_arr.append(train(network, train_loader, optimizer, device).detach().cpu())
    acc_arr.append(test(network, test_loader, device, i))
    scheduler.step()


final_test(network, test_loader, device)
save_model(network, TEST_NUM, epochs_arr, loss_arr, acc_arr)
