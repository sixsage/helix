import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

SEED = 172
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.8
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.001
EPOCH = 150

RESULTS_PATH = 'autoencoder_points\\results\\'
DATASET_PATH = 'tracks_1m.txt'

class Dataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'r') as file:
            content = file.read()
        data_points = content.split('EOT')

        data_points = [dp.strip() for dp in data_points if dp.strip()]
        data_points = [dp.split('\n') for dp in data_points]
        data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
        self.targets = torch.tensor(np.array([dp[0] for dp in data_points]))
        input_points = [dp[1:] for dp in data_points]
        self.inputs = torch.tensor(np.array(input_points))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        return input
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 12)
        self.layer2 = nn.Linear(12, 48)
        self.layer3 = nn.Linear(48, 192)
        self.layer4 = nn.Linear(192, 6)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x
    
class Decoder(nn.Module):

    def __init__(self):
        self.layer1 = nn.Linear(6, 192)
        self.layer2 = nn.Linear(192, 48)
        self.layer3 = nn.Linear(48, 12)
        self.layer4 = nn.Linear(12, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 12)
        self.layer2 = nn.Linear(12, 48)
        self.layer3 = nn.Linear(48, 192)
        self.layer4 = nn.Linear(192, 6)
        self.layer5 = nn.Linear(6, 192)
        self.layer6 = nn.Linear(192, 48)
        self.layer7 = nn.Linear(48, 12)
        self.layer8 = nn.Linear(12, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = self.layer7(x)
        x = self.layer8(x)
        return x
    
def train_model(model, criterion, optimizer, scheduler, train_dl, val_dl, device, results_file, num_epochs=EPOCH, prev_model_path=''):
    if prev_model_path and os.path.isfile(prev_model_path):
        print(f"Loading model from {prev_model_path}")
        checkpoint = torch.load(prev_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No model path provided, starting training from scratch")
        start_epoch = 1
        with open(results_file, 'a') as file:  # Open the results file in append mode
            file.write("Epoch,Train Loss,Val Loss\n")  # Write the header
    
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_losses = []

        bar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for points in bar:
            points = points.float().to(device)

            optimizer.zero_grad()
            output = model(points)
            loss = criterion(output, points)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            bar.set_postfix(loss=np.mean(train_losses))

        avg_train_loss = np.mean(train_losses)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for points in val_dl:
                points = points.float().to(device)

                output = model(points)
                val_loss = criterion(output, points)

                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        # Write epoch results to file
        with open(results_file, 'a') as file:
            file.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        # save model
        if epoch % 5 == 0:
            save_path = RESULTS_PATH + f'pointnet_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'scheduler': scheduler.state_dict()
            }, save_path)
            print(f"Model saved to {save_path} after epoch {epoch}")
        scheduler.step()    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = Dataset(path=DATASET_PATH)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = Autoencoder()
    model = model.to(device)

    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50], gamma=0.5)
    
    train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCH, train_dl=train_dataloader, val_dl=val_dataloader, device=device, results_file=RESULTS_PATH+'results.csv')