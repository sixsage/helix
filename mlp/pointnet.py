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
# LEARNING_RATE = 0.001 # original
EPOCH = 170

RESULTS_PATH = 'mlp\\1m_pointnet_4sigfig\\'
DATASET_PATH = 'tracks_1m_updated.txt'

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
        target = self.targets[idx]
        input = self.inputs[idx]
        return input, target

# for i, dp in enumerate(data_points):
#     print(f"Data Point {i+1}:\n{dp}\n")

class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 1024)
        self.layer4 = nn.Linear(1024, 512)
        self.layer5 = nn.Linear(512, 256)
        self.layer6 = nn.Linear(256, 5)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.max(x, 1)[0]
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        result = x
        # result = torch.zeros_like(x)
        # result[0] = F.relu(x[0])
        # result[1] = torch.sigmoid(x[1]) * 6.28
        # result[2] = torch.sigmoid(x[2]) * 175 + 25
        # result[3] = x[3]
        # result[4] = x[4]
        return result
    
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
        for input, target in bar:
            input = input.float().to(device)
            target = target.float().to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            bar.set_postfix(loss=np.mean(train_losses))

        avg_train_loss = np.mean(train_losses)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for input, target in val_dl:
                input = input.float().to(device)
                target = target.float().to(device)

                output = model(input)
                val_loss = criterion(output, target)

                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        # Write epoch results to file
        with open(results_file, 'a') as file:
            file.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        # save model
        if epoch >= 100 and epoch % 5 == 0:
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

    model = PointNet()
    model = model.to(device)
    if torch.cuda.is_available():
        model.cuda()
        unet_model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 30, 50], gamma=0.5)
    
    train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCH, train_dl=train_dataloader, val_dl=val_dataloader, device=device, results_file=RESULTS_PATH+'results.csv',)