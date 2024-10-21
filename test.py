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
    
class test():
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
                break
            self.xy_coordinates = torch.tensor(np.array(self.xy_coordinates))
            self.z_coordinates = torch.tensor(np.array(self.z_coordinates))
            self.xyz_coordinates = torch.tensor(np.array(self.xyz_coordinates))
            print(self.xy_coordinates.size())


if __name__ == '__main__':
    test(path=VAL_DATASET_PATH)