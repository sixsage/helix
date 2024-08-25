import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import scipy.optimize

SEED = 172
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.8
BATCH_SIZE = 8
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.001 # original
EPOCH = 170

RESULTS_PATH = 'mlp_point_regression\\1m_pointnet\\'
DATASET_PATH = 'tracks.txt'

MIN_R0=1.0
MAX_R0=10.0
N_LAYERS=10
SIGMA=0.01

FIND_PHI_ITERATION = 1000

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
    
# from: https://www-jlc.kek.jp/subg/offl/lib/docs/helix_manip/node3.html
# a track is parameterized by a 5dim helix
# here it is expressed relative to an initial angle, phi0, and calculated and some updated angle phi
# hence it sweeps out the track as you vary phi and keep the other parameters fixed
def track(phi, d0,phi0,pt,dz,tanl):
    alpha = 1/2 # 1/cB
    q=1
    kappa = q/pt
    rho = alpha/kappa
    x = d0*np.cos(phi0) + rho*(np.cos(phi0)-np.cos(phi0+phi))
    y = d0*np.sin(phi0) + rho*(np.sin(phi0)-np.sin(phi0+phi))
    z = dz - rho*tanl*phi
    return x,y,z

# For a given phi and track parameters, calculates the distance from a target r02
# used by find_phi to determine the angle that intersects with a fixed-radius circle detector
def dr(phi, r02,d0,phi0,pt,dz,tanl):

    # get the xyz of the track at this phi
    x,y,z = track(phi, d0,phi0,pt,dz,tanl)
    r2=x*x+y*y

    # get the distance from the target r02
    dr = np.fabs(r2-r02)

    return dr

# Find the phi value where the track intersects a given layer r
def find_phi(r0, d0,phi0,pt,dz,tanl):

    # this is lazy, but rather than inverting the equations we just minimize the distance
    res = scipy.optimize.minimize(dr,0,method='Nelder-Mead',args = (r0, d0,phi0,pt,dz,tanl))#, bounds =(0,1.0))

    return res.x[0]

# calculate the chisq between the track defined by the parameters and the spacepoints given
def chisq(params,x,y,z):
    ihit = 0
    c2=0
    length = params.shape[0]
    # find the hits for the track parameters
    print(length)
    for i in range(length):
        ihit = 0
        param = params[i]
        target_x = x[i]
        target_y = y[i]
        target_z = z[i]
        for r0 in np.linspace(MIN_R0,MAX_R0,N_LAYERS):
            phi0 = find_phi(r0*r0,*param)
            x0,y0,z0 = track(phi0,*param)
            # calculate deviation from observed hit
            c2 = c2 + (x0-target_x[ihit])**2 + (y0-target_y[ihit])**2 + (z0-target_z[ihit])**2   # assume equal uncertainty in x,y,z
            ihit = ihit+1

    return c2 / length

def track_torch(phi, d0, phi0, pt, dz, tanl):
    alpha = 1/2 # 1/cB
    q = 1
    kappa = q/pt
    rho = alpha/kappa
    x = d0 * torch.cos(phi0) + rho * (torch.cos(phi0) - torch.cos(phi0 + phi))
    y = d0 * torch.sin(phi0) + rho * (torch.sin(phi0) - torch.sin(phi0 + phi))
    z = dz - rho * tanl * phi
    return x, y, z

def dr_torch(phi, r02, d0, phi0, pt, dz, tanl):
    # get the xyz of the track at this phi
    x, y, z = track_torch(phi, d0, phi0, pt, dz, tanl)
    r2 = x * x + y * y

    # get the distance from the target r02
    dr = torch.abs(r2 - r02)
    return dr.mean()

# Find the phi value where the track intersects a given layer r
def find_phi_torch(r0, d0, phi0, pt, dz, tanl):
    # this is lazy, but rather than inverting the equations we just minimize the distance
    phi = torch.tensor([0.0], requires_grad=True, device='cuda')
    optimizer = torch.optim.SGD([phi], lr=0.01)

    for i in range(FIND_PHI_ITERATION):
        print(i)
        optimizer.zero_grad()
        loss = dr_torch(phi, r0, d0, phi0, pt, dz, tanl)
        if i + 1 == FIND_PHI_ITERATION:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
        optimizer.step()

    return phi.item()

def chisq_torch(params, x, y, z):
    ihit = 0
    c2 = 0
    # find the hits for the track parameters
    for r0 in torch.linspace(MIN_R0, MAX_R0, N_LAYERS):
        phi0 = find_phi_torch(r0 * r0, params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4])
        x0, y0, z0 = track_torch(phi0, params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4])
        print(x0)
        print(x[:, ihit])
        # calculate deviation from observed hit
        c2 += (x0 - x[:, ihit])**2 + (y0 - y[:, ihit])**2 + (z0 - z[:, ihit])**2   # assume equal uncertainty in x,y,z
        ihit += 1

    return c2
    
def MSE_for_points(output, target):
    print(output.size())
    print(target.size())
    print(chisq_torch(output, target[:, :, 0], target[:, :, 1], target[:, :, 2]))
    
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
            loss = criterion(output, input)
            return
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

    criterion = MSE_for_points
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 30, 50], gamma=0.5)
    
    train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCH, train_dl=train_dataloader, val_dl=val_dataloader, device=device, results_file=RESULTS_PATH+'results.csv',)