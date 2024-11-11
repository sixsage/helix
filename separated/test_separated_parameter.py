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
Z_LEARNING_RATE = 0.001
# LEARNING_RATE = 0.001
EPOCH = 50

# DATASET_PATH = 'tracks_100k_updated_asymmetric_higher.txt'
DATASET_PATH = 'tracks_100k_updated_non_gaussian_wider.txt'
ENCODER_PATH = 'separated\\not_gaussian\\results4\\encoder_epoch_250.pth'

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

            # split targets into targets for (x, y) and (z)
            self.rescaled_targets_xy = self.rescaled_targets[:, [0, 1, 4, 5]]
            self.rescaled_targets_z = self.rescaled_targets[:, 1:4]

            self.xy_coordinates = []
            self.z_coordinates = []
            self.xyz_coordinates = []
            for input in input_points:
                combined_xy = []
                combined_z = []
                combined = []
                for x, y, z in input:
                    combined_xy.append(x)
                    combined_xy.append(y)

                    combined_z.append(z)

                    combined.append(x)
                    combined.append(y)
                    combined.append(z)
                self.xy_coordinates.append(combined_xy)
                self.z_coordinates.append(combined_z)
                self.xyz_coordinates.append(combined)
            self.xy_coordinates = torch.tensor(np.array(self.xy_coordinates), dtype=torch.float32)
            self.z_coordinates = torch.tensor(np.array(self.z_coordinates), dtype=torch.float32)
            self.xyz_coordinates = torch.tensor(np.array(self.xyz_coordinates), dtype=torch.float32)
            # self.inputs = torch.tensor(np.array(input_points))

    def __len__(self):
        return len(self.rescaled_targets)

    def __getitem__(self, idx):
        input_xy = self.xy_coordinates[idx]
        input_z = self.z_coordinates[idx]
        target = self.original_targets[idx]
        return input_xy, input_z, target
    
class MSEWithTrigConstraint(nn.Module):
    def __init__(self, trig_lagrangian=1, device='cuda'):
        super().__init__()
        self.trig_lagrangian = trig_lagrangian
        self.mse_loss = nn.MSELoss()
        self.device = device
    
    def forward(self, prediction, target):
        total_loss = self.mse_loss(prediction, target)
        cos_values = prediction[:, -2]
        sin_values = prediction[:, -1]
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
        self.layer1_xy = nn.Linear(20, 200)
        self.layer2_xy = nn.Linear(200, 400)
        self.layer3_xy = nn.Linear(400, 800)
        self.layer4_xy = nn.Linear(800, 800)
        self.layer5_xy = nn.Linear(800, 800)
        self.layer6_xy = nn.Linear(800, 400)
        self.layer7_xy = nn.Linear(400, 200)
        self.output_layer_xy = nn.Linear(200, 4)
        self.xy_encoder = nn.Sequential(
            self.layer1_xy,
            nn.LeakyReLU(inplace=True),
            self.layer2_xy,
            nn.LeakyReLU(inplace=True),
            self.layer3_xy,
            nn.LeakyReLU(inplace=True),
            self.layer4_xy,
            nn.LeakyReLU(inplace=True),
            self.layer5_xy,
            nn.LeakyReLU(inplace=True),
            self.layer6_xy,
            nn.LeakyReLU(inplace=True),
            self.layer7_xy,
            nn.LeakyReLU(inplace=True),
            self.output_layer_xy
        )

        self.layer1_z = nn.Linear(10, 200)
        self.layer2_z = nn.Linear(200, 400)
        self.layer3_z = nn.Linear(400, 800)
        self.layer4_z = nn.Linear(800, 800)
        self.layer5_z = nn.Linear(800, 400)
        self.layer6_z = nn.Linear(400, 200)
        self.output_layer_z = nn.Linear(200, 3)

        self.z_encoder = nn.Sequential(
            self.layer1_z,
            nn.LeakyReLU(inplace=True),
            self.layer2_z,
            nn.LeakyReLU(inplace=True),
            self.layer3_z,
            nn.LeakyReLU(inplace=True),
            self.layer4_z,
            nn.LeakyReLU(inplace=True),
            self.layer5_z,
            nn.LeakyReLU(inplace=True),
            self.layer6_z,
            nn.LeakyReLU(inplace=True),
            self.output_layer_z
        )


    def forward(self, xy_input, z_input):
        # xy = F.leaky_relu(self.layer1_xy(xy_input))
        # xy = F.leaky_relu(self.layer2_xy(xy))
        # xy = F.leaky_relu(self.layer3_xy(xy))
        # xy = F.leaky_relu(self.layer4_xy(xy))
        # xy = F.leaky_relu(self.layer5_xy(xy))
        # xy = F.leaky_relu(self.layer6_xy(xy))
        # xy = F.leaky_relu(self.layer7_xy(xy))
        # xy = self.output_layer_xy(xy)

        # z = F.leaky_relu(self.layer1_z(z_input))
        # z = F.leaky_relu(self.layer2_z(z))
        # z = F.leaky_relu(self.layer3_z(z))
        # z = F.leaky_relu(self.layer4_z(z))
        # z = F.leaky_relu(self.layer5_z(z))
        # z = F.leaky_relu(self.layer6_z(z))
        # z = self.output_layer_z(z)
        xy = self.xy_encoder(xy_input)
        z = self.z_encoder(z_input)
        return xy, z
    
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
        for input_xy, input_z, targets in dataloader:
            input_xy = input_xy.to(device)
            input_z = input_z.to(device)
            targets = targets.to(device)

            outputs_xy, outputs_z = encoder(input_xy, input_z)
            output_xy_part1 = outputs_xy[:, :2]  
            output_xy_part2 = outputs_xy[:, 2:] 
            outputs = torch.cat((output_xy_part1, outputs_z[:, 1:], output_xy_part2), dim=1).to(device)
            outputs = torch.Tensor.cpu(outputs)
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
    encoder_optimizer = torch.optim.Adam([
        {'params': encoder.xy_encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': encoder.z_encoder.parameters(), 'lr': Z_LEARNING_RATE, 'weight_decay': 1e-5}
    ], lr = LEARNING_RATE)
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[15, 30, 50], gamma=0.5)
    
    test_parameters(encoder, encoder_optimizer, dataset, device, prev_encoder_path=ENCODER_PATH)