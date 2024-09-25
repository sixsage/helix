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
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.001
EPOCH = 50

DATASET_PATH = 'tracks_1m_updated.txt'
ENCODER_RESULTS_PATH = 'autoencoder_points\\results_trig8_minmax\\'
DECODER_RESULTS_PATH = 'autoencoder_points\\results_trig8_minmax\\'
    
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
            inputs = []
            for input in input_points:
                combined = []
                for coordinate in input:
                    combined += coordinate
                inputs.append(combined)
            self.inputs = torch.tensor(np.array(inputs))
            # self.inputs = torch.tensor(np.array(input_points))

    def __len__(self):
        return len(self.rescaled_targets)

    def __getitem__(self, idx):
        target = self.rescaled_targets[idx]
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
            file.write("Epoch,Train Loss,Val Loss\n")  # Write the header

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
    
    for epoch in range(start_epoch, num_epochs + 1):
        encoder.train()
        train_losses_encoder = []

        bar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for points, params in bar:
            points = points.float().to(device)
            params = params.float().to(device)

            encoder_optimizer.zero_grad()
            output = encoder(points)
            output_copy = output.detach().clone().requires_grad_()
            encoder_loss = encoder_criterion(output, params)
            encoder_loss.backward()
            encoder_optimizer.step()

            train_losses_encoder.append(encoder_loss.item())
            bar.set_postfix(loss=np.mean(train_losses_encoder))

        avg_train_loss_encoder = np.mean(train_losses_encoder)

        encoder.eval()
        with torch.no_grad():
            val_losses_encoder = []
            for points, params in val_dl:
                points = points.float().to(device)
                params = params.float().to(device)

                output = encoder(points)
                val_loss = encoder_criterion(output, params)

                val_losses_encoder.append(val_loss.item())

        avg_val_loss_encoder = np.mean(val_losses_encoder)

        # Write epoch results to file
        with open(results_file_encoder, 'a') as file:
            file.write(f"{epoch},{avg_train_loss_encoder:.9f},{avg_val_loss_encoder:.9f}\n")

        # save model
        if epoch % 5 == 0:
            save_path = ENCODER_RESULTS_PATH + f'encoder_epoch_{epoch}.pth'
            if encoder_scheduler:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': encoder_optimizer.state_dict(),
                    'loss': encoder_loss.item(),
                    'scheduler': encoder_scheduler.state_dict()
                }, save_path)
            else:
                 torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': encoder_optimizer.state_dict(),
                    'loss': encoder_loss.item()
                }, save_path)
            print(f"Encoder saved to {save_path} after epoch {epoch}")
        if encoder_scheduler and decoder_scheduler:
            encoder_scheduler.step()

    for epoch in range(start_epoch, num_epochs + 1):
        decoder.train()
        train_losses_decoder = []

        bar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for points, params in bar:
            points = points.float().to(device)
            params = params.float().to(device)

            encoder_optimizer.zero_grad()
            output = encoder(points)
            output_copy = output.detach().clone().requires_grad_()

            decoder_optimizer.zero_grad()
            reconstructed_points = decoder(output_copy)
            decoder_loss = decoder_criterion(reconstructed_points, points)
            decoder_loss.backward()
            decoder_optimizer.step()

            train_losses_decoder.append(decoder_loss.item())
            bar.set_postfix(loss=np.mean(train_losses_decoder))

        avg_train_loss_decoder = np.mean(train_losses_decoder)

        decoder.eval()
        with torch.no_grad():
            val_losses_decoder = []
            for points, params in val_dl:
                points = points.float().to(device)
                params = params.float().to(device)

                output = encoder(points)

                reconstructed_points = decoder(output)
                val_loss_decoder = decoder_criterion(reconstructed_points, points)

                val_losses_decoder.append(val_loss_decoder.item())

        avg_val_loss_decoder = np.mean(val_losses_decoder)

        with open(results_file_decoder, 'a') as file:
            file.write(f"{epoch},{avg_train_loss_decoder:.9f},{avg_val_loss_decoder:.9f}\n")

        # save model
        if epoch % 5 == 0:
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = Dataset(path=DATASET_PATH)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[10, 20, 30], gamma=0.5)
    decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[10, 20, 30], gamma=0.5)
    # encoder_scheduler = None
    # decoder_scheduler = None
    
    train_encoder_decoder(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, 
                          num_epochs=EPOCH, train_dl=train_dataloader, val_dl=val_dataloader, device=device, 
                          results_file_encoder=ENCODER_RESULTS_PATH+'encoder_results.csv', results_file_decoder=DECODER_RESULTS_PATH+"decoder_results.csv")