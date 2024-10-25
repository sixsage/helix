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
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
# LEARNING_RATE = 0.001
EPOCH = 250
REPORT_PATH = 'experiments.txt'

# non_gaussian_wider, regular gaussian - lr = 0.0001, epoch = 100, batch 512, mileston: none
# asymmetric - lr = 0.001, epoch = 50, batch 512, milestone 10, 20, 30

DATASET_PATH = 'tracks_1m_updated_non_gaussian_wider.txt'
VAL_DATASET_PATH = 'tracks_100k_updated_non_gaussian_wider.txt'
ENCODER_RESULTS_PATH = 'separated\\not_gaussian\\results3\\'
DECODER_RESULTS_PATH = 'separated\\not_gaussian\\results3\\'

# DATASET_PATH = 'tracks_1m_updated.txt'
# VAL_DATASET_PATH = 'tracks_100k_updated.txt'
# ENCODER_RESULTS_PATH = 'separated\\gaussian\\results2_gaussian\\'
# DECODER_RESULTS_PATH = 'separated\\gaussian\\results2_gaussian\\'

# DATASET_PATH = 'tracks_1m_updated_asymmetric_higher.txt'
# VAL_DATASET_PATH = 'tracks_100k_updated_asymmetric_higher.txt'
# ENCODER_RESULTS_PATH = 'separated\\asymmetric\\results1\\'
# DECODER_RESULTS_PATH = 'separated\\asymmetric\\results1\\'
    
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
            self.rescaled_targets = torch.tensor(self.rescaled_targets)
            self.original_targets = torch.tensor(targets_cos_sin)

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
            self.xy_coordinates = torch.tensor(np.array(self.xy_coordinates))
            self.z_coordinates = torch.tensor(np.array(self.z_coordinates))
            self.xyz_coordinates = torch.tensor(np.array(self.xyz_coordinates))
            # self.inputs = torch.tensor(np.array(input_points))

    def __len__(self):
        return len(self.rescaled_targets)

    def __getitem__(self, idx):
        target_xy = self.rescaled_targets_xy[idx]
        target_z = self.rescaled_targets_z[idx]
        input_xy = self.xy_coordinates[idx]
        input_z = self.z_coordinates[idx]
        input_xyz = self.xyz_coordinates[idx]
        return input_xy, input_z, input_xyz, target_xy, target_z
    
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

def train_encoder_decoder(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, num_epochs, train_dl, val_dl, device, results_file_encoder='', results_file_decoder='', prev_encoder_path ='', prev_decoder_path=''):
    if prev_encoder_path and os.path.isfile(prev_encoder_path):
        print(f"Loading model from {prev_encoder_path}")
        checkpoint = torch.load(prev_encoder_path)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No model path provided, starting training from scratch")
        start_epoch = 1
        with open(results_file_encoder, 'a') as file:  # Open the results file in append mode
            file.write("Epoch,Train Loss XY, Val Loss XY, Train Loss Z, Val Loss Z\n")  # Write the header

    if prev_decoder_path and os.path.isfile(prev_decoder_path):
        print(f"Loading model from {prev_decoder_path}")
        checkpoint = torch.load(prev_decoder_path)
        decoder.load_state_dict(checkpoint['model_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No model path provided, starting training from scratch")
        start_epoch = 1
        with open(results_file_decoder, 'a') as file:  # Open the results file in append mode
            file.write("Epoch,Train Loss,Val Loss\n")  # Write the header

    mseLoss = nn.MSELoss()
    
    for epoch in range(start_epoch, num_epochs + 1):
        encoder.train()
        train_losses_encoder_xy = []
        train_losses_encoder_z = []

        bar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for input_xy, input_z, _, target_xy, target_z in bar:
            input_xy = input_xy.float().to(device)
            input_z = input_z.float().to(device)
            target_xy = target_xy.float().to(device)
            target_z = target_z.float().to(device)

            encoder_optimizer.zero_grad()
            output_xy, output_z = encoder(input_xy, input_z)
            encoder_loss_xy = encoder_criterion(output_xy, target_xy)
            encoder_loss_xy.backward()
            encoder_loss_z = mseLoss(output_z, target_z)
            encoder_loss_z.backward()
            encoder_optimizer.step()

            train_losses_encoder_xy.append(encoder_loss_xy.item())
            train_losses_encoder_z.append(encoder_loss_z.item())
            bar.set_postfix(loss=np.mean(train_losses_encoder_xy))

        avg_train_loss_encoder_xy = np.mean(train_losses_encoder_xy)
        avg_train_loss_encoder_z = np.mean(train_losses_encoder_z)

        encoder.eval()
        with torch.no_grad():
            val_losses_encoder_xy = []
            val_losses_encoder_z = []
            for input_xy, input_z, _, target_xy, target_z in val_dl:
                input_xy = input_xy.float().to(device)
                input_z = input_z.float().to(device)
                target_xy = target_xy.float().to(device)
                target_z = target_z.float().to(device)

                output_xy, output_z = encoder(input_xy, input_z)
                val_loss_xy = encoder_criterion(output_xy, target_xy)
                val_loss_z = mseLoss(output_z, target_z)

                val_losses_encoder_xy.append(val_loss_xy.item())
                val_losses_encoder_z.append(val_loss_z.item())

        avg_val_loss_encoder_xy = np.mean(val_losses_encoder_xy)
        avg_val_loss_encoder_z = np.mean(val_losses_encoder_z)

        # Write epoch results to file
        with open(results_file_encoder, 'a') as file:
            file.write(f"{epoch},{avg_train_loss_encoder_xy:.9f},{avg_val_loss_encoder_xy:.9f},{avg_train_loss_encoder_z:.9f},{avg_val_loss_encoder_z:.9f}\n")

        # save model
        if epoch >= (EPOCH - 20) and epoch % 5 == 0:
            save_path = ENCODER_RESULTS_PATH + f'encoder_epoch_{epoch}.pth'
            if encoder_scheduler:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': encoder_optimizer.state_dict(),
                    'loss': encoder_loss_xy.item(),
                    'scheduler': encoder_scheduler.state_dict()
                }, save_path)
            else:
                 torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': encoder_optimizer.state_dict(),
                    'loss': encoder_loss_xy.item()
                }, save_path)
            print(f"Encoder saved to {save_path} after epoch {epoch}")
        if encoder_scheduler and decoder_scheduler:
            encoder_scheduler.step()

    for epoch in range(start_epoch, num_epochs + 1):
        decoder.train()
        train_losses_decoder = []

        bar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for input_xy, input_z, input_xyz, target_xy, target_z in bar:
            input_xy = input_xy.float().to(device)
            input_z = input_z.float().to(device)
            input_xyz = input_xyz.float().to(device)
            target_xy = target_xy.float().to(device)
            target_z = target_z.float().to(device)


            encoder_optimizer.zero_grad()
            output_xy, output_z = encoder(input_xy, input_z)
            output_xy_part1 = output_xy[:, :2]  
            output_xy_part2 = output_xy[:, 2:] 

            encoder_output = torch.cat((output_xy_part1, output_z[:, 1:], output_xy_part2), dim=1).to(device)
            
            decoder_optimizer.zero_grad()
            reconstructed_points = decoder(encoder_output)
            decoder_loss = decoder_criterion(reconstructed_points, input_xyz)
            decoder_loss.backward()
            decoder_optimizer.step()

            train_losses_decoder.append(decoder_loss.item())
            bar.set_postfix(loss=np.mean(train_losses_decoder))

        avg_train_loss_decoder = np.mean(train_losses_decoder)

        decoder.eval()
        with torch.no_grad():
            val_losses_decoder = []
            for input_xy, input_z, input_xyz, target_xy, target_z in val_dl:
                input_xy = input_xy.float().to(device)
                input_z = input_z.float().to(device)
                input_xyz = input_xyz.float().to(device)
                target_xy = target_xy.float().to(device)
                target_z = target_z.float().to(device)

                output_xy, output_z = encoder(input_xy, input_z)
                output_xy_part1 = output_xy[:, :2]  
                output_xy_part2 = output_xy[:, 2:]
                encoder_output = torch.cat((output_xy_part1, output_z[:, 1:], output_xy_part2), dim=1).to(device)

                reconstructed_points = decoder(encoder_output)
                val_loss_decoder = decoder_criterion(reconstructed_points, input_xyz)

                val_losses_decoder.append(val_loss_decoder.item())

        avg_val_loss_decoder = np.mean(val_losses_decoder)

        with open(results_file_decoder, 'a') as file:
            file.write(f"{epoch},{avg_train_loss_decoder:.9f},{avg_val_loss_decoder:.9f}\n")

        # save model
        if epoch >= ( EPOCH - 20) and epoch % 5 == 0:
            save_path = DECODER_RESULTS_PATH + f'decoder_epoch_{epoch}.pth'
            if decoder_scheduler:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': decoder_loss.item(),
                    'scheduler': decoder_scheduler.state_dict()
                }, save_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': decoder_loss.item()
                }, save_path)
            print(f"Decoder saved to {save_path} after epoch {epoch}")
        if encoder_scheduler and decoder_scheduler:
            decoder_scheduler.step()

def writeTrainingInfo(encoder_scheduler, decoder_scheduler):
    s = f'''For: {ENCODER_RESULTS_PATH}
# of epochs: {EPOCH}
learning rate: {LEARNING_RATE}
batch size: {BATCH_SIZE}
encoder milestones: {encoder_scheduler.milestones if encoder_scheduler else "None"}
decoder milestones: {decoder_scheduler.milestones if decoder_scheduler else "None"}\n\n'''
    with open(REPORT_PATH, 'a') as file:
        file.write(s)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = Dataset(path=DATASET_PATH)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset = Dataset(path=VAL_DATASET_PATH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    encoder = Encoder()
    encoder = encoder.to(device)
    decoder = Decoder()
    decoder = decoder.to(device)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    encoder_criterion = MSEWithTrigConstraint(trig_lagrangian=0.001)
    decoder_criterion = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = LEARNING_RATE)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = LEARNING_RATE)
    # encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[15, 40], gamma=0.1)
    # decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[15, 40], gamma=0.1)
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[100], gamma=0.5)
    decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[100], gamma=0.5)
    # encoder_scheduler = None
    # decoder_scheduler = None
    
    train_encoder_decoder(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, 
                          num_epochs=EPOCH, train_dl=train_dataloader, val_dl=val_dataloader, device=device, 
                          results_file_encoder=ENCODER_RESULTS_PATH+'encoder_results.csv', results_file_decoder=DECODER_RESULTS_PATH+"decoder_results.csv")
    
    writeTrainingInfo(encoder_scheduler, decoder_scheduler)