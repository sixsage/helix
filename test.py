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
EPOCH = 120
REPORT_PATH = 'experiments.txt'

# non_gaussian_wider, regular gaussian - lr = 0.0001, epoch = 100, batch 512, mileston: none
# asymmetric - lr = 0.001, epoch = 50, batch 512, milestone 10, 20, 30

# DATASET_PATH = 'tracks_1m_updated_non_gaussian_wider.txt'
# VAL_DATASET_PATH = 'tracks_100k_updated_non_gaussian_wider.txt'
# ENCODER_RESULTS_PATH = 'not_gaussian\\results6_trig_wider_minmax\\'
# DECODER_RESULTS_PATH = 'not_gaussian\\results6_trig_wider_minmax\\'

DATASET_PATH = 'tracks_1m_updated.txt'
VAL_DATASET_PATH = 'tracks_100k_updated.txt'
ENCODER_RESULTS_PATH = 'autoencoder_points\\results_trig13_minmax\\'
DECODER_RESULTS_PATH = 'autoencoder_points\\results_trig13_minmax\\'

# DATASET_PATH = 'tracks_1m_updated_asymmetric_higher.txt'
# VAL_DATASET_PATH = 'tracks_100k_updated_asymmetric_higher.txt'
# ENCODER_RESULTS_PATH = 'asymmetric\\results4_higher_trig_minmax\\'
# DECODER_RESULTS_PATH = 'asymmetric\\results4_higher_trig_minmax\\'

min_r0=1.0
max_r0=10.0
nlayers=10
sigma=0.01

def track_torch(phi, params):
    # Unpack parameters from the tensor
    d0, phi0, pt, dz, tanl = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
    
    alpha = 0.5  # equivalent to 1/cB
    q = 1
    kappa = q / pt
    rho = alpha / kappa
    
    x = d0 * torch.cos(phi0) + rho * (torch.cos(phi0) - torch.cos(phi0 + phi))
    y = d0 * torch.sin(phi0) + rho * (torch.sin(phi0) - torch.sin(phi0 + phi))
    z = dz - rho * tanl * phi
    
    return x.float(), y.float(), z.float()

def fast_find_phi_torch(r02, params):
    # Initialize values for batch processing
    ra2 = torch.zeros(params.shape[0], device=params.device)
    phia = torch.zeros(params.shape[0], device=params.device)
    phib = torch.full((params.shape[0],), 0.1, device=params.device)
    
    xb, yb, zb = track_torch(phib, params)
    rb2 = xb * xb + yb * yb
    
    finished_phia = torch.zeros(params.shape[0], device=params.device)
    updated = torch.zeros(params.shape[0], device=params.device).bool()
    # Iteratively update values
    while torch.any(updated == False):
        mask1 = ((rb2 > r02) & (ra2 < r02))
        mask2 = ((rb2 < r02) & (ra2 < r02))
        mask3 = ((rb2 > r02) & (ra2 > r02))
        
        # Update where condition (rb2 > r02 and ra2 < r02)
        phib[mask1] = phia[mask1] + ((phib[mask1] - phia[mask1]) * (r02[mask1] - ra2[mask1]) / (rb2[mask1] - ra2[mask1]))
        xb[mask1], yb[mask1], zb[mask1] = track_torch(phib[mask1], params[mask1])
        rb2[mask1] = xb[mask1] * xb[mask1] + yb[mask1] * yb[mask1]
        
        # Update where condition (rb2 < r02 and ra2 < r02)
        phib_new = phia[mask2] + (phib[mask2] - phia[mask2]) * (r02[mask2] - ra2[mask2]) / (rb2[mask2] - ra2[mask2])
        phia[mask2] = phib[mask2]
        ra2[mask2] = rb2[mask2]
        phib[mask2] = phib_new
        xb[mask2], yb[mask2], zb[mask2] = track_torch(phib[mask2], params[mask2])
        rb2[mask2] = xb[mask2] * xb[mask2] + yb[mask2] * yb[mask2]
        
        # Update where condition (rb2 > r02 and ra2 > r02)
        phia_new = phib[mask3] + (phia[mask3] - phib[mask3]) * (r02[mask3] - rb2[mask3]) / (ra2[mask3] - rb2[mask3])
        phib[mask3] = phia[mask3]
        rb2[mask3] = ra2[mask3]
        phia[mask3] = phia_new
        xa, ya, za = track_torch(phia[mask3], params[mask3])
        # print(xa.shape)
        ra2[mask3] = xa * xa + ya * ya

        # Copy into finsihed
        condition = (rb2 - ra2 <= 0.01)
        finished_phia[condition] = phia[condition]
        updated[condition] = True
        print(rb2[updated == False])
        print(ra2[updated == False])
        print(r02[updated == False])
        print('while')
    
    return finished_phia

def track(phi, d0,phi0,pt,dz,tanl):
    alpha = 1/2 # 1/cB
    q=1
    kappa = q/pt
    rho = alpha/kappa
    x = d0*np.cos(phi0) + rho*(np.cos(phi0)-np.cos(phi0+phi))
    y = d0*np.sin(phi0) + rho*(np.sin(phi0)-np.sin(phi0+phi))
    z = dz - rho*tanl*phi
    return x,y,z

def fast_find_phi(r02, d0,phi0,pt,dz,tanl):

    ra2 = 0
    phia=0
    phib = 0.1
    xb,yb,zb= track(phib,d0,phi0,pt,dz,tanl)
    rb2 = xb*xb+yb*yb

    while (rb2-ra2>0.01):
        if (rb2>r02 and ra2<r02):
            phib = phia + (phib-phia)*(r02-ra2)/(rb2-ra2)
            xb,yb,zb= track(phib,d0,phi0,pt,dz,tanl)
            rb2 = xb*xb+yb*yb
        if (rb2<r02 and ra2<r02):
            phibnew = phia + (phib-phia)*(r02-ra2)/(rb2-ra2)
            phia = phib
            ra2 = rb2
            phib = phibnew
            xb,yb,zb= track(phib,d0,phi0,pt,dz,tanl)
            rb2 = xb*xb+yb*yb
        if (rb2>r02 and ra2>r02):
            phianew = phib + (phia-phib)*(r02-rb2)/(ra2-rb2)
            phib = phia
            rb2 = ra2
            phia = phianew
            xa,ya,za= track(phia,d0,phi0,pt,dz,tanl)
            ra2 = xa*xa+ya*ya
        print('while')
    print('end')
    return phia
    
class test():
    def __init__(self, path, transform=None):
        with open(path, 'r') as file:
            content = file.read()
            data_points = content.split('EOT')[:3000]

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
            self.original_targets_cos_sin = torch.tensor(targets_cos_sin)
            self.original_targets = torch.tensor(self.original_targets)

            # split targets into targets for (x, y) and (z)
            # self.rescaled_targets_xy = self.rescaled_targets[:, [0, 1, 4, 5]]
            # self.rescaled_targets_z = self.rescaled_targets[:, 1:4]

            # self.xy_coordinates = []
            # self.z_coordinates = []
            # self.xyz_coordinates = []
            # for input in input_points:
            #     combined_xy = []
            #     combined_z = []
            #     combined = []
            #     for x, y, z in input:
            #         combined_xy.append(x)
            #         combined_xy.append(y)

            #         combined_z.append(z)

            #         combined.append(x)
            #         combined.append(y)
            #         combined.append(z)
            #     self.xy_coordinates.append(combined_xy)
            #     self.z_coordinates.append(combined_z)
            #     self.xyz_coordinates.append(combined)
            #     break
            # self.xy_coordinates = torch.tensor(np.array(self.xy_coordinates))
            # self.z_coordinates = torch.tensor(np.array(self.z_coordinates))
            # self.xyz_coordinates = torch.tensor(np.array(self.xyz_coordinates))
            # print(self.xy_coordinates.size())
            inputs = []
            for input in input_points:
                combined = []
                for coordinate in input:
                    combined += coordinate
                inputs.append(combined)
            self.inputs = torch.tensor(np.array(inputs))

    def __len__(self):
        return len(self.original_targets)

    def __getitem__(self, idx):
        param = self.original_targets[idx]
        return param


if __name__ == '__main__':
    # train_dataset = test(path=VAL_DATASET_PATH)
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # for params in train_dataloader:
    #     for r0 in np.linspace(min_r0,max_r0,nlayers):
    #         print(r0)
    #         r0_tensor = torch.full((params.shape[0],), r0)
    #         phi0 = fast_find_phi_torch(r0_tensor * r0_tensor, params)
    #         x, y, z = track_torch(phi0, params)
    #     print(x)
    #     print(y)
        # params2 = params.tolist()[0]
        # xs= []
        # ys= []
        # zs = []
        # for r0 in np.linspace(min_r0,max_r0,nlayers):
        #     phi0 = fast_find_phi(r0 * r0, *params2)
        #     x, y, z = track(phi0, *params2)
        #     xs.append(x)
        #     ys.append(y)
        #     zs.append(z)
        # print(xs)
        # print(ys)
        
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randn(3, 5)
    # mse_none = torch.nn.MSELoss(reduction='none')
    # mse_mean = torch.nn.MSELoss()
    # print(mse_none(input, target))
    # print(torch.mean(mse_none(input, target)))
    # print(mse_mean(input, target))
    qwe = torch.tensor([1, 2, 3, 4, 5, 6])
    asd = torch.tensor([[2, 3, 7, 6, 3, 0]])
    zxc = torch.tensor([[6, 11, 6, 4, 5, 10]])
    qwe = torch.zeros((3, 6))
    asd = torch.tensor([[2, 3, 7, 6, 3, 0]])
    zxc = torch.tensor([
        [2, 3, 7, 6, 3, 0],
        [2, 3, 7, 10, 3, 0],
        [2, 1, 9, 6, 3, 0]
    ])

    # mask1 = ((zxc - asd > 0) & (zxc - asd < 5))
    # print(qwe)
    # print(mask1)
    # print(qwe[mask1])
    # qwe[mask1] = asd[mask1]
    # print(qwe)
    # print(torch.any(zxc == 0))
    # print(qwe)
    # print(zxc)
    # squared_diff = (qwe - zxc) ** 2
    # result = squared_diff.sum(dim=-1).sum(dim=-1, keepdim= True).flatten()
    # print(result)
    # torch.save(result, 'temp_save\\test.pt')
    # print(torch.load('temp_save\\test.pt'))
    print(torch.mean(zxc.float(), dim=1))