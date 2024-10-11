import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.8
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.001
EPOCH = 50

DATASET_PATH = 'tracks_100k_updated.txt'
ENCODER_PATH = 'autoencoder_points\\results_trig10_minmax\\encoder_epoch_100.pth'
    
class Dataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'r') as file:
            content = file.read()
            data_points = content.split('EOT')

            data_points = [dp.strip() for dp in data_points if dp.strip()]
            data_points = [dp.split('\n') for dp in data_points]
            data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
            self.original_targets = np.array([dp[0] for dp in data_points])
            input_points = [dp[1:] for dp in data_points]
            targets_2 = np.delete(self.original_targets, 1, 1)
            targets_2 = np.hstack((targets_2, np.cos(self.original_targets[:, 1])[..., None]))
            targets_cos_sin = np.hstack((targets_2, np.sin(self.original_targets[:, 1])[..., None]))
            self.scaler = MinMaxScaler()
            self.rescaled_targets = self.scaler.fit_transform(targets_cos_sin)
            self.rescaled_targets = torch.tensor(self.rescaled_targets, dtype=torch.float32)
            self.original_targets = torch.tensor(targets_cos_sin, dtype=torch.float32)
            inputs = []
            for input in input_points:
                combined = []
                for coordinate in input:
                    combined += coordinate
                inputs.append(combined)
            self.inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
            # self.inputs = torch.tensor(np.array(input_points))

    def __len__(self):
        return len(self.rescaled_targets)

    def __getitem__(self, idx):
        target = self.original_targets[idx]
        input = self.inputs[idx]
        return input, target
    
class MSEWithTrigConstraint(nn.Module):
    def __init__(self, trig_lagrangian=1, device='cuda'):
        super().__init__()
        self.trig_lagrangian = trig_lagrangian
        self.mse_loss = nn.MSELoss()
        self.device = device
    
    def forward(self, prediction, target):
        total_loss = self.mse_loss(prediction, target)
        cos_values = prediction[:, 4]
        sin_values = prediction[:, 5]
        norm = torch.sqrt(torch.square(cos_values) + torch.square(sin_values)).to(self.device)
        ones = torch.ones(*norm.size()).to(self.device)
        constraint_loss = torch.abs(ones - norm)
        total_loss += self.trig_lagrangian * torch.mean(constraint_loss)
        return total_loss
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = nn.Linear(30, 32)
        # self.layer2 = nn.Linear(32, 64)
        # self.layer3 = nn.Linear(64, 5)
        self.layer1 = nn.Linear(30, 200)
        self.layer2 = nn.Linear(200, 400)
        self.layer3 = nn.Linear(400, 800)
        self.layer4 = nn.Linear(800, 800)
        self.layer5 = nn.Linear(800, 800)
        self.layer6 = nn.Linear(800, 400)
        self.layer7 = nn.Linear(400, 200)
        self.output_layer = nn.Linear(200, 6)


    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = F.leaky_relu(self.layer4(x))
        x = F.leaky_relu(self.layer5(x))
        x = F.leaky_relu(self.layer6(x))
        x = F.leaky_relu(self.layer7(x))
        x = self.output_layer(x)
        return x
    
class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        # self.layer1 = nn.Linear(5, 64)
        # self.layer2 = nn.Linear(64, 32)
        # self.layer3 = nn.Linear(32, 30)
        self.layer1 = nn.Linear(6, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 200)
        self.layer4 = nn.Linear(200, 400)
        self.layer5 = nn.Linear(400, 800)
        self.layer6 = nn.Linear(800, 800)
        self.layer7 = nn.Linear(800, 800)
        self.layer8 = nn.Linear(800, 400)
        self.layer9 = nn.Linear(400, 200)
        self.output_layer = nn.Linear(200, 30)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = self.output_layer(x)
        return x
    
def absolute_difference(prediction, actual):
    return torch.abs(actual - prediction)

def change_trig_to_angle(input):
    cos_values = input[:, 4]
    sin_values = input[:, 5]
    pred_angle = torch.atan2(sin_values, cos_values)
    pred_angle= torch.remainder(pred_angle, np.pi).reshape(input.shape[0], 1)
    return torch.hstack((input[:, 0:1], pred_angle, input[:, 1:4]))

def test_parameters(encoder, encoder_optimizer, dataset, device, prev_encoder_path):
    print(f"Loading model from {prev_encoder_path}")
    checkpoint = torch.load(prev_encoder_path)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    mae_losses = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = torch.Tensor.cpu(encoder(inputs))
            outputs = dataset.scaler.inverse_transform(outputs)
            predictions_with_sin_cos = torch.tensor(outputs).to(device)
            predictions_with_angle = change_trig_to_angle(predictions_with_sin_cos)
            targets_with_angle = change_trig_to_angle(targets)
            mae_losses.append(absolute_difference(predictions_with_angle, targets_with_angle))

    result_mae = torch.cat(mae_losses)
    print(result_mae.size())
    result_mae = torch.mean(result_mae, dim=0).to(device)
    print('before dividing')
    print(result_mae)
    original_targets_with_angle = change_trig_to_angle(dataset.original_targets)
    min_per_column = torch.min(original_targets_with_angle, dim=0, keepdim=True)
    max_per_column = torch.max(original_targets_with_angle, dim=0, keepdim=True)
    range_per_column = (max_per_column.values - min_per_column.values).to(device)
    # (0,0.02),(0,2*np.pi),(25,200),(-2.5,2.5),(-1.0,1.0))
    # ranges = torch.tensor([0.02, 2 * np.pi, 175, 5, 2]).to(device)
    print('ranges per column')
    print(range_per_column)
    print('mae / range in percentage')
    print(torch.div(result_mae, range_per_column) * 100)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = Dataset(DATASET_PATH)

    encoder = Encoder()
    encoder = encoder.to(device)

    if torch.cuda.is_available():
        encoder.cuda()
    encoder_criterion = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = LEARNING_RATE)
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[15, 30, 50], gamma=0.5)
    
    test_parameters(encoder, encoder_optimizer, dataset, device, prev_encoder_path=ENCODER_PATH)