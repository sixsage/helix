import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

SEED = 172
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.8
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
EPOCH = 80

INPUT_SIZE = 30
TRANSFORMER_INPUT_SIZE = 128
NUM_HEADS = 8
NUM_LAYERS = 6
OUTPUT_SIZE = 5

RESULTS_PATH = 'mlp\\1m_mlp_ryan\\'
MODEL_PATH = 'mlp_epoch_20.pth'
DATASET_PATH = 'tracks_1m.txt'

class MLP_Dataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'r') as file:
            content = file.read()
        data_points = content.split('EOT')

        data_points = [dp.strip() for dp in data_points if dp.strip()]
        data_points = [dp.split('\n') for dp in data_points]
        data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
        targets = np.array([dp[0] for dp in data_points])
        self.scaler = MinMaxScaler()
        self.rescaled_targets = self.scaler.fit_transform(targets)
        self.rescaled_targets = torch.tensor(self.rescaled_targets, dtype=torch.float32)
        self.original_targets = torch.tensor(targets, dtype=torch.float32)
        input_points = [dp[1:] for dp in data_points]
        inputs = []
        for input in input_points:
            combined = []
            for coordinate in input:
                combined += coordinate
            inputs.append(combined)
        self.inputs = torch.tensor(np.array(inputs), dtype=torch.float32)

    def __len__(self):
        return len(self.original_targets)

    def __getitem__(self, idx):
        target = self.original_targets[idx]
        input = self.inputs[idx]
        return input, target
    
class PointNet_Dataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'r') as file:
            content = file.read()
        data_points = content.split('EOT')

        data_points = [dp.strip() for dp in data_points if dp.strip()]
        data_points = [dp.split('\n') for dp in data_points]
        data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
        self.targets = torch.tensor(np.array([dp[0] for dp in data_points]), dtype=torch.float32)
        input_points = [dp[1:] for dp in data_points]
        self.inputs = torch.tensor(np.array(input_points), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target = self.targets[idx]
        input = self.inputs[idx]
        return input, target

# for i, dp in enumerate(data_points):
#     print(f"Data Point {i+1}:\n{dp}\n")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(30, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 5)
        )
    def forward(self, x):
        return self.layers(x)
    
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
        return x
    

INDEX = 6

def absolute_difference(prediction, actual):
    return torch.abs(actual - prediction)

def percentage_difference(prediction, actual, epsilon=0.0001):
    return 100 * torch.abs(prediction - actual) / (torch.abs(actual) + epsilon)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = MLP_Dataset(path=DATASET_PATH)
    # dataset = MLP_Dataset(path=DATASET_PATH)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # model = MLP()
    model = MLP()
    model = model.to(device)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 25], gamma=0.5)

    prev_model_path = RESULTS_PATH + MODEL_PATH
    print(f"Loading model from {prev_model_path}")
    checkpoint = torch.load(prev_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    scheduler.load_state_dict(checkpoint['scheduler'])

    # test_input = None
    # test_target = None
    # test_output = None
    mae_losses = []
    percentage_differences = []
    INDEX = 8
    
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = torch.Tensor.cpu(model(inputs))
            outputs = dataset.scaler.inverse_transform(outputs)
            outputs = torch.tensor(outputs).to(device)
            mae_losses.append(absolute_difference(outputs, targets))
            percentage_differences.append(percentage_difference(outputs, targets))
            # interest_actual = targets[INDEX]
            # interest_prediction = outputs[INDEX]
            
    result_mae = torch.cat(mae_losses)
    print(result_mae.size())
    result_mae = torch.mean(result_mae, dim=0).to(device)
    print('before dividing')
    print(result_mae)
    min_per_column = torch.min(dataset.original_targets, dim=0, keepdim=True)
    max_per_column = torch.max(dataset.original_targets, dim=0, keepdim=True)
    range_per_column = (max_per_column.values - min_per_column.values).to(device)
    # (0,0.02),(0,2*np.pi),(25,200),(-2.5,2.5),(-1.0,1.0))
    # ranges = torch.tensor([0.02, 2 * np.pi, 175, 5, 2]).to(device)
    print('ranges per column')
    print(range_per_column)
    print('mae / range in percentage')
    print(torch.div(result_mae, range_per_column) * 100)

    result_percentage = torch.cat(percentage_differences)
    result_percentage = torch.mean(result_percentage, dim=0)
    print('mape')
    print(result_percentage)
    # plot_helix_interactive(test_target, test_output, test_input)

    # plot_helix_static(test_target, test_output, test_input, view_angle='top')

    # # Plot side view
    # plot_helix_static(test_target, test_output, test_input, view_angle='side')