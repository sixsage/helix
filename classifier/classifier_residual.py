import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

min_r0=1.0
max_r0=10.0
nlayers=10
sigma=0.01

SEED = 172
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.8
BATCH_SIZE = 1
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.001 # original
EPOCH = 170

MODEL_PATH = "mlp\\1m_pointnet_second\\pointnet_epoch_170.pth"
HELIX_PATH = 'tracks.txt'
NON_HELIX_PATH = 'sintracks_10000.txt'

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
    ihit=0
    c2=0
    # find the hits for the track parameters
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0 = find_phi(r0*r0,*params)
        x0,y0,z0 = track(phi0,*params)

        # calculate deviation from observed hit
        c2 = c2 + (x0-x[ihit])**2 + (y0-y[ihit])**2 + (z0-z[ihit])**2   # assume equal uncertainty in x,y,z
        ihit = ihit+1

    return c2


# calculate the track parameters for a set of spacepoints (0,0.69,100.,0,0.1)
def fit_params(x,y,z):

    res = scipy.optimize.minimize(chisq,(0,0.69,100.,0,0.1),args=(x,y,z),method='Nelder-Mead', bounds = ( (0,0.02),(0,2*np.pi),(25,200),(-2.5,2.5),(-1.0,1.0)) )
    return res.x


# find the intersections with the detector layers for these track parameters, add noise
def make_hits(params):
    xs=[]
    ys=[]
    zs =[]
    
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0 = find_phi(r0*r0,*params)
        x0,y0,z0 = track(phi0,*params)
        xs.append(x0+np.random.normal(scale=sigma))
        ys.append(y0+np.random.normal(scale=sigma))
        zs.append(z0+np.random.normal(scale=sigma))


    return xs,ys,zs

class Dataset(Dataset):
    def __init__(self, helix_path, non_helix_path, size, transform=None):
        with open(helix_path, 'r') as file:
            helix_content = file.read()
        helix_data_points = helix_content.split('EOT')[:size]

        helix_data_points = [dp.strip() for dp in helix_data_points if dp.strip()]
        helix_data_points = [dp.split('\n') for dp in helix_data_points]
        helix_data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in helix_data_points]
        helix_input_points = [dp[1:] for dp in helix_data_points]
        self.helix_inputs = torch.tensor(np.array(helix_input_points))

        with open(non_helix_path, 'r') as file:
            non_helix_content = file.read()
        non_helix_data_points = non_helix_content.split('EOT')[:size]

        non_helix_data_points = [dp.strip() for dp in non_helix_data_points if dp.strip()]
        non_helix_data_points = [dp.split('\n') for dp in non_helix_data_points]
        non_helix_data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in non_helix_data_points]
        non_helix_input_points = [dp[1:] for dp in non_helix_data_points]
        self.non_helix_inputs = torch.tensor(np.array(non_helix_input_points))
        self.combined = torch.cat((self.helix_inputs, self.non_helix_inputs), dim=0)
        self.change = self.helix_inputs.size()[0]

    def __len__(self):
        return self.combined.size()[0]

    def __getitem__(self, idx):
        target = 1 if idx < self.change else 0
        input = self.combined[idx]
        return input, target

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
        return result
    
def calculate_distance(parameters, input):
    input = input[0]
    x = input[:, 0]
    y = input[:, 1]
    z = input[:, 2]

    return chisq(parameters, x, y, z)

def test_residual(model, criterion, optimizer, scheduler, val_dl, device, prev_model_path, threshold=1):
    print(f"Loading model from {prev_model_path}")
    checkpoint = torch.load(prev_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    with torch.no_grad():
        predictions = None
        targets = None
        distances = []
        for input, target in val_dl:
            input = input.float().to(device)
            target = target.float().to(device)

            output_parameters = torch.squeeze(model(input))
            numpy_input = input.cpu().numpy()
            distance = calculate_distance(tuple(output_parameters.cpu().tolist()), numpy_input)
            distances.append(distance)
            if distance > threshold:
                output = torch.zeros((1)).to(device)
            else:
                output = torch.ones((1)).to(device)

            if predictions == None:
                predictions = output
                targets = target
            else:
                predictions = torch.cat((predictions, output), 0)
                targets = torch.cat((targets, target), 0)

    print(distances[-1], distances[-2], distances[-3])
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

    dataset = Dataset(HELIX_PATH, NON_HELIX_PATH, size=1000)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = PointNet()
    model = model.to(device)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
    
    test_residual(model, criterion, optimizer, scheduler, val_dl=dataloader, device=device, prev_model_path=MODEL_PATH, threshold=20)









