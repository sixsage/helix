import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

TRAIN_RATIO = 0.8
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.001 # original
EPOCH = 150

# MODEL_PATH = "autoencoder_points\\results2\\autoencoder_epoch_40.pth"
ENCODER_PATH = "autoencoder_points\\results5\\encoder_epoch_55.pth"
DECODER_PATH = "autoencoder_points\\results5\\decoder_epoch_55.pth"
HELIX_PATH = 'tracks_100k_4sigfig.txt'
NON_HELIX_PATH = 'sintracks_100k_4sigfig.txt'

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
        self.output_layer = nn.Linear(200, 5)


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
        self.layer1 = nn.Linear(5, 200)
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
        for input, target in val_dl:
            input = input.float().to(device)
            target = target.float().to(device)

            output_points = decoder(encoder(input))
            first_dim = torch.numel(output_points)  // 30
            output_points = output_points.reshape(first_dim, 10, 3)
            reshaped_points = input.reshape(first_dim, 10, 3)
            distance = torch.norm(output_points - reshaped_points, 2, dim=(1, 2))
            distances[current_size:current_size + distance.shape[0]] = distance
            output = (distance < threshold).float()
            predictions[current_size:current_size + output.shape[0]] = output
            targets[current_size:current_size + target.shape[0]] = target
            current_size += distance.shape[0]

    print(distances[-1])
    print(distances)
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
                  prev_encoder_path=ENCODER_PATH, prev_decoder_path=DECODER_PATH, data_size=len(dataset), threshold=10)