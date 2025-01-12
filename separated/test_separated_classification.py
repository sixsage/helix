import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.8
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
# LEARNING_RATE = 0.001
EPOCH = 300
REPORT_PATH = 'experiments.txt'

ENCODER_PATH = 'separated\\asymmetric\\results1\\encoder_epoch_300.pth'
DECODER_PATH = 'separated\\asymmetric\\results1\\decoder_epoch_300.pth'
HELIX_PATH = 'tracks_100k_updated_asymmetric_higher_test.txt'
NON_HELIX_PATH = 'sintracks_100k_updated_asymmetric_higher_test.txt'
THRESHOLD = 2

# ENCODER_PATH = 'separated\\not_gaussian\\results2\\encoder_epoch_300.pth'
# DECODER_PATH = 'separated\\not_gaussian\\results2\\decoder_epoch_300.pth'
# HELIX_PATH = 'tracks_100k_updated_non_gaussian_wider.txt'
# NON_HELIX_PATH = 'sintracks_100k_updated_non_gaussian_wider.txt'
# THRESHOLD = 3.5

# ENCODER_PATH = 'separated\\results1_gaussian\\encoder_epoch_300.pth'
# DECODER_PATH = 'separated\\results1_gaussian\\decoder_epoch_300.pth'
# HELIX_PATH = 'tracks_100k_updated_test.txt'
# NON_HELIX_PATH = 'sintracks_100k_updated_test.txt'
# THRESHOLD = 2
    
class Dataset(Dataset):
    def __init__(self, helix_path, non_helix_path, transform=None):
        with open(helix_path, 'r') as file:
            content = file.read()
            helix_data_points = content.split('EOT')

            helix_data_points = [dp.strip() for dp in helix_data_points if dp.strip()]
            helix_data_points = [dp.split('\n') for dp in helix_data_points]
            helix_data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in helix_data_points]
            self.original_helix_targets = np.array([dp[0] for dp in helix_data_points])
            input_points = [dp[1:] for dp in helix_data_points]
            targets_2 = np.delete(self.original_helix_targets, 1, 1)
            targets_2 = np.hstack((targets_2, np.cos(self.original_helix_targets[:, 1])[..., None]))
            self.helix_targets_cos_sin = torch.tensor(np.hstack((targets_2, np.sin(self.original_helix_targets[:, 1])[..., None]))) # targets are not updated to get separate for xy and z

            self.xy_coordinates_helical = []
            self.z_coordinates_helical = []
            self.xyz_coordinates_helical = []
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
                self.xy_coordinates_helical.append(combined_xy)
                self.z_coordinates_helical.append(combined_z)
                self.xyz_coordinates_helical.append(combined)
            self.xy_coordinates_helical = torch.tensor(np.array(self.xy_coordinates_helical))
            self.z_coordinates_helical = torch.tensor(np.array(self.z_coordinates_helical))
            self.xyz_coordinates_helical = torch.tensor(np.array(self.xyz_coordinates_helical))
            # self.inputs = torch.tensor(np.array(input_points))

        with open(non_helix_path, 'r') as file:
            non_helix_content = file.read()
        non_helix_data_points = non_helix_content.split('EOT')

        non_helix_data_points = [dp.strip() for dp in non_helix_data_points if dp.strip()]
        non_helix_data_points = [dp.split('\n') for dp in non_helix_data_points]
        non_helix_data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in non_helix_data_points]
        non_helix_input_points = [dp[1:] for dp in non_helix_data_points]

        self.xy_coordinates_non_helical = []
        self.z_coordinates_non_helical = []
        self.xyz_coordinates_non_helical = []
        for helix_input in non_helix_input_points:
            combined_xy = []
            combined_z = []
            combined = []
            for x, y, z in helix_input:
                combined_xy.append(x)
                combined_xy.append(y)

                combined_z.append(z)

                combined.append(x)
                combined.append(y)
                combined.append(z)

            self.xy_coordinates_non_helical.append(combined_xy)
            self.z_coordinates_non_helical.append(combined_z)
            self.xyz_coordinates_non_helical.append(combined)

        self.xy_coordinates_non_helical = torch.tensor(np.array(self.xy_coordinates_non_helical))
        self.z_coordinates_non_helical = torch.tensor(np.array(self.z_coordinates_non_helical))
        self.xyz_coordinates_non_helical = torch.tensor(np.array(self.xyz_coordinates_non_helical))

        self.combined_xyz = torch.cat((self.xyz_coordinates_helical, self.xyz_coordinates_non_helical), dim=0)
        self.combined_xy = torch.cat((self.xy_coordinates_helical, self.xy_coordinates_non_helical), dim=0)
        self.combined_z = torch.cat((self.z_coordinates_helical, self.z_coordinates_non_helical), dim=0)
        self.change = self.xyz_coordinates_helical.size()[0]

    def __len__(self):
        return self.combined_xyz.size()[0]

    def __getitem__(self, idx):
        target = 1 if idx < self.change else 0
        input_xy = self.combined_xy[idx]
        input_z = self.combined_z[idx]
        input_xyz = self.combined_xyz[idx]
        return input_xy, input_z, input_xyz, target
    
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

        self.layer1_z = nn.Linear(10, 200)
        self.layer2_z = nn.Linear(200, 400)
        self.layer3_z = nn.Linear(400, 800)
        self.layer4_z = nn.Linear(800, 800)
        self.layer5_z = nn.Linear(800, 400)
        self.layer6_z = nn.Linear(400, 200)
        self.output_layer_z = nn.Linear(200, 3)


    def forward(self, xy_input, z_input):
        xy = F.leaky_relu(self.layer1_xy(xy_input))
        xy = F.leaky_relu(self.layer2_xy(xy))
        xy = F.leaky_relu(self.layer3_xy(xy))
        xy = F.leaky_relu(self.layer4_xy(xy))
        xy = F.leaky_relu(self.layer5_xy(xy))
        xy = F.leaky_relu(self.layer6_xy(xy))
        xy = F.leaky_relu(self.layer7_xy(xy))
        xy = self.output_layer_xy(xy)

        z = F.leaky_relu(self.layer1_z(z_input))
        z = F.leaky_relu(self.layer2_z(z))
        z = F.leaky_relu(self.layer3_z(z))
        z = F.leaky_relu(self.layer4_z(z))
        z = F.leaky_relu(self.layer5_z(z))
        z = F.leaky_relu(self.layer6_z(z))
        z = self.output_layer_z(z)
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
    
def test_distance(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, val_dl, device, prev_encoder_path, prev_decoder_path, data_size, threshold=1):
    print(f"Loading model from {prev_encoder_path}")
    checkpoint = torch.load(prev_encoder_path)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loading model from {prev_decoder_path}")
    checkpoint = torch.load(prev_decoder_path)
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    with torch.no_grad():
        predictions = torch.zeros(data_size)
        targets = torch.zeros(data_size)
        distances = torch.zeros(data_size)
        current_size = 0
        for input_xy, input_z, input_xyz, target in val_dl:
            input_xy = input_xy.float().to(device)
            input_z = input_z.float().to(device)
            input_xyz = input_xyz.float().to(device)

            encoder_output_xy, encoder_output_z = encoder(input_xy, input_z)
            output_xy_part1 = encoder_output_xy[:, :2]  
            output_xy_part2 = encoder_output_xy[:, 2:]
            encoder_output = torch.cat((output_xy_part1, encoder_output_z[:, 1:], output_xy_part2), dim=1).to(device)

            output_points = decoder(encoder_output)
            first_dim = torch.numel(output_points)  // 30
            output_points = output_points.reshape(first_dim, 10, 3)
            reshaped_points = input_xyz.reshape(first_dim, 10, 3)
            squared_diff = (output_points - reshaped_points) ** 2
            pairwise_distances = torch.sqrt(squared_diff.sum(dim=-1))
            distance = pairwise_distances.sum(dim=-1)
            # distance = torch.norm(output_points - reshaped_points, 2, dim=(1, 2))
            # distance = torch.sum(torch.square(output_points - reshaped_points), dim=(1, 2))
            distances[current_size:current_size + distance.shape[0]] = distance
            output = (distance < threshold).float()
            predictions[current_size:current_size + output.shape[0]] = output
            targets[current_size:current_size + target.shape[0]] = target
            current_size += distance.shape[0]

    print(distances[-1])
    print(distances)
    print(distances.shape)
    print(torch.mean(distances))
    print("predictions:", predictions)
    print("targets:", targets)
    print("predicted helix but was actually non-helix")
    predict_helix = predictions == 1
    wrong = predictions != targets
    mask = torch.logical_and(predict_helix, wrong)
    print(predictions[mask].size())
    print("predicted non-helix but was helix")
    predict_non_helix = predictions == 0
    wrong = predictions != targets
    mask = torch.logical_and(predict_non_helix, wrong)
    print(predictions[mask].size())
    print((predictions == targets).float().mean().item())

    # sanity check
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    print(accuracy_score(targets_np, predictions_np))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = Dataset(HELIX_PATH, NON_HELIX_PATH)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    encoder = Encoder()
    encoder = encoder.to(device)
    decoder = Decoder()
    decoder = decoder.to(device)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    encoder_criterion = nn.MSELoss()
    decoder_criterion = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = LEARNING_RATE)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = LEARNING_RATE)
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[15, 30, 50], gamma=0.5)
    decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[15, 30, 50], gamma=0.1)
    
    test_distance(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, val_dl=dataloader, device=device, 
                  prev_encoder_path=ENCODER_PATH, prev_decoder_path=DECODER_PATH, data_size=len(dataset), threshold=THRESHOLD)