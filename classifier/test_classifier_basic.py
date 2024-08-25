import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics

SEED = 172
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.8
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.001 # original
EPOCH = 40

RESULTS_PATH = 'classifier\\classifier_basic\\'
HELIX_PATH = 'tracks.txt'
NON_HELIX_PATH = 'sintracks_10000.txt'
INPUT_SIZE = 30

class Dataset(Dataset):
    def __init__(self, helix_path, non_helix_path, transform=None):
        with open(helix_path, 'r') as file:
            helix_content = file.read()
        helix_data_points = helix_content.split('EOT')

        helix_data_points = [dp.strip() for dp in helix_data_points if dp.strip()]
        helix_data_points = [dp.split('\n') for dp in helix_data_points]
        helix_data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in helix_data_points]
        helix_input_points = [dp[1:] for dp in helix_data_points]
        helix_inputs = []
        for helix_input in helix_input_points:
            combined = []
            for coordinate in helix_input:
                combined += coordinate
            helix_inputs.append(combined)
        self.helix_inputs = torch.tensor(np.array(helix_inputs))

        with open(non_helix_path, 'r') as file:
            non_helix_content = file.read()
        non_helix_data_points = non_helix_content.split('EOT')

        non_helix_data_points = [dp.strip() for dp in non_helix_data_points if dp.strip()]
        non_helix_data_points = [dp.split('\n') for dp in non_helix_data_points]
        non_helix_data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in non_helix_data_points]
        non_helix_input_points = [dp[1:] for dp in non_helix_data_points]
        non_helix_inputs = []
        for helix_input in non_helix_input_points:
            combined = []
            for coordinate in helix_input:
                combined += coordinate
            non_helix_inputs.append(combined)
        self.non_helix_inputs = torch.tensor(np.array(non_helix_inputs))
        self.combined = torch.cat((self.helix_inputs, self.non_helix_inputs), dim=0)
        self.change = self.helix_inputs.size()[0]

    def __len__(self):
        return self.combined.size()[0]

    def __getitem__(self, idx):
        target = 1 if idx < self.change else 0
        input = self.combined[idx]
        return input, target
    
class MLP_Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 128)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 64)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    
def run_model(model, criterion, optimizer, scheduler, val_dl, device, prev_model_path):
    print(f"Loading model from {prev_model_path}")
    checkpoint = torch.load(prev_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    with torch.no_grad():
        predictions = None
        targets = None
        for input, target in val_dl:
            input = input.float().to(device)
            target = target.float().to(device)
            target = torch.unsqueeze(target, dim=1)

            output = model(input)
            acc = (output.round() == target).float().mean().item()

            if predictions == None:
                predictions = output
                targets = target
            else:
                predictions = torch.cat((predictions, output), 0)
                targets = torch.cat((targets, target), 0)
    print((predictions.round() == targets).float().mean().item())
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    predictions = np.concatenate(predictions).ravel()
    targets = np.concatenate(targets).ravel()
    
    fpr, tpr, thresholds = metrics.roc_curve(targets, predictions)
    # print(fpr)
    # print(tpr)
    print(metrics.auc(fpr, tpr))



        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = Dataset(HELIX_PATH, NON_HELIX_PATH)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MLP_Classifier(INPUT_SIZE)
    model = model.to(device)
    if torch.cuda.is_available():
        model.cuda()
        unet_model = nn.DataParallel(model)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
    
    run_model(model, criterion, optimizer, scheduler, val_dl=val_dataloader, device=device, prev_model_path=RESULTS_PATH+'classify_basic_epoch_40.pth')

