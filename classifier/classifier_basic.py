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
            file.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n")  # Write the header
    
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_losses = []
        train_accuracies = []
        bar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for input, target in bar:
            input = input.float().to(device)
            target = target.float().to(device)
            target = torch.unsqueeze(target, dim=1)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            acc = (output.round() == target).float().mean().item()
            train_losses.append(loss.item())
            train_accuracies.append(acc)
            bar.set_postfix(loss=np.mean(train_losses), acc=float(acc))

        avg_train_loss = np.mean(train_losses)
        avg_train_accuracy = np.mean(train_accuracies)

        model.eval()
        with torch.no_grad():
            val_losses = []
            val_accuracies = []
            for input, target in val_dl:
                input = input.float().to(device)
                target = target.float().to(device)
                target = torch.unsqueeze(target, dim=1)

                output = model(input)
                val_loss = criterion(output, target)
                acc = (output.round() == target).float().mean().item()

                val_losses.append(val_loss.item())
                val_accuracies.append(acc)

        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)

        # Write epoch results to file
        with open(results_file, 'a') as file:
            file.write(f"{epoch},{avg_train_loss:.4f},{avg_train_accuracy:.4f},{avg_val_loss:.4f},{avg_val_accuracy:.4f}\n")

        # save model
        if epoch % 5 == 0:
            save_path = RESULTS_PATH + f'classify_basic_epoch_{epoch}.pth'
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
    
    train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCH, train_dl=train_dataloader, val_dl=val_dataloader, device=device, results_file=RESULTS_PATH+'results.csv')
    
# def test(helix_path, non_helix_path, transform=None):
#         with open(helix_path, 'r') as file:
#             helix_content = file.read()
#         helix_data_points = helix_content.split('EOT')

#         helix_data_points = [dp.strip() for dp in helix_data_points if dp.strip()]
#         helix_data_points = [dp.split('\n') for dp in helix_data_points]
#         helix_data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in helix_data_points]
#         helix_input_points = [dp[1:] for dp in helix_data_points]
#         helix_inputs = []
#         for helix_input in helix_input_points:
#             combined = []
#             for coordinate in helix_input:
#                 combined += coordinate
#             helix_inputs.append(combined)
#         helix_inputs = torch.tensor(np.array(helix_inputs))
#         print(helix_inputs[0])
#         ones = torch.ones(helix_inputs.size()[0], 1)
#         print(torch.cat((helix_inputs, ones), 1))

#         with open(non_helix_path, 'r') as file:
#             non_helix_content = file.read()
#         non_helix_data_points = non_helix_content.split('EOT')

#         non_helix_data_points = [dp.strip() for dp in non_helix_data_points if dp.strip()]
#         non_helix_data_points = [dp.split('\n') for dp in non_helix_data_points]
#         non_helix_data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in non_helix_data_points]
#         non_helix_input_points = [dp[1:] for dp in non_helix_data_points]
#         non_helix_inputs = []
#         for helix_input in non_helix_input_points:
#             combined = []
#             for coordinate in helix_input:
#                 combined += coordinate
#             non_helix_inputs.append(combined)
#         non_helix_inputs = torch.tensor(np.array(non_helix_inputs))
#         print(non_helix_inputs[0])
#         zeros = torch.zeros(non_helix_inputs.size()[0], 1)
#         print(torch.cat((non_helix_inputs, zeros), 1))
#         combined = torch.cat((helix_inputs, non_helix_inputs), dim=0)
#         print(combined[1])

# test(HELIX_PATH, NON_HELIX_PATH)

