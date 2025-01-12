import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import warnings

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.8
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
# LEARNING_RATE = 0.001
ENCODER_EPOCH = 100
DECODER_EPOCH = 0
REPORT_PATH = 'experiments.txt'

LAYER_ARR = [200, 400, 800, 1600, 1600, 1600, 800, 400, 400, 400, 200, 100]
KLD_WEIGHT = 1e-6


# mixture
# DATASET_PATH = 'non_gaussian_tracks\\tracks_1M_gaussian_mixturev2.txt'
# VAL_DATASET_PATH = 'non_gaussian_tracks\\tracks_100k_gaussian_mixturev2.txt'
# ENCODER_RESULTS_PATH = 'variance_predict\\mixture\\results3\\'
# DECODER_RESULTS_PATH = 'variance_predict\\mixture\\results3\\'

DATASET_PATH = 'tracks_1M_updated.txt'
VAL_DATASET_PATH = 'tracks_100k_updated.txt'
ENCODER_RESULTS_PATH = 'variance_predict\\gaussian\\vae\\vae_1_var_results1\\'
DECODER_RESULTS_PATH = 'variance_predict\\gaussian\\vae_results1\\'

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
            self.scaler = MinMaxScaler()
            self.original_targets_rescaled = self.scaler.fit_transform(self.original_targets)
            self.original_targets_tensor = torch.tensor(self.original_targets)
            inputs = []
            for input in input_points:
                combined = []
                for coordinate in input:
                    combined += coordinate
                inputs.append(combined)
            self.inputs = torch.tensor(np.array(inputs))
            # self.inputs = torch.tensor(np.array(input_points))

    def __len__(self):
        return len(self.original_targets_rescaled)

    def __getitem__(self, idx):
        target = self.original_targets_tensor[idx]
        input = self.inputs[idx]
        return input, target
    
def find_phi_inverse(target_radius_squared, d0, phi0, pt, dz, tanl, eps=1e-6):
    alpha = 1 / 2  # 1/cB
    q = 1
    kappa = q / pt
    rho = alpha / kappa

    calculated_term_x = d0 * torch.cos(phi0) + rho * torch.cos(phi0) 
    calculated_term_y = d0 * torch.sin(phi0) + rho * torch.sin(phi0)

    hypotenuse = torch.sqrt(calculated_term_x ** 2 + calculated_term_y ** 2)

    arccos_input = (calculated_term_x ** 2 + calculated_term_y ** 2 + rho ** 2 - target_radius_squared) / (2 * rho * hypotenuse)
    clamped = torch.clamp(arccos_input, min=-1 + eps, max=1-eps) # to avoid nan
    arccos_term = torch.acos(clamped)
    phi = arccos_term   

    return phi % (2 * torch.pi) # wrap angle

def compute_3d_track(phi, d0, phi0, pt, dz, tanl):
    alpha = 1 / 2  # 1/cB
    q = 1
    kappa = q / pt
    rho = alpha / kappa
    x = d0 * torch.cos(phi0) + rho * (torch.cos(phi0) - torch.cos(phi0 + phi))
    y = d0 * torch.sin(phi0) + rho * (torch.sin(phi0) - torch.sin(phi0 + phi))
    z = dz - rho * tanl * phi
    return x, y, z
    
# Generate noiseless hit points in 3D using optimized phi values
def generate_noiseless_hits(params, min_radius=1.0, max_radius=10.0, num_layers=10):

    d0, phi0, pt, dz, tanl = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
    xs, ys, zs = [], [], []
    target_radii = [torch.tensor(r, requires_grad=True) for r in torch.linspace(min_radius, max_radius, num_layers)]
    #for target_radius in torch.linspace(min_radius, max_radius, num_layers, device=device):
    for target_radius in target_radii:
        # Optimize phi for the current radius
        phi = find_phi_inverse(target_radius**2, d0, phi0, pt, dz, tanl)
        x, y, z = compute_3d_track(phi, d0, phi0, pt, dz, tanl)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Stack results into a tensor
    hit_points = torch.stack((torch.stack(xs, dim=1), torch.stack(ys, dim=1), torch.stack(zs, dim=1)), dim=2)

    #return hit_points, total_distance / num_layers, torch.stack(optimized_phi_values)
    return hit_points

class GenerateNoiselessHitsLayer(nn.Module):
    def __init__(self):
        super(GenerateNoiselessHitsLayer, self).__init__()

    def forward(self, z):
        return generate_noiseless_hits(z)

class VAE_Encoder(nn.Module):
    def __init__(self, input_size=30, hidden_sizes=LAYER_ARR, latent_size_mean=5, latent_size_var=5):
        super().__init__()
        layer_sizes = [input_size] + hidden_sizes
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)])
        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_size_mean)
        self.fc_logvar = nn.Linear(hidden_sizes[-1], latent_size_var)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        # print("x input was ", x)
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        # print("mu before was ", mu[0])
        # print("logvar before was ", logvar[0])
        z = self.reparameterize(mu, logvar)
        # print("z after reparameterize ", z[0])
        return z, mu, logvar


class VAE_EncoderWithHits(nn.Module):
    def __init__(self, input_size=30, hidden_sizes=LAYER_ARR, latent_size_mean=5, latent_size_var=5):
        super().__init__()
        self.encoder = VAE_Encoder(input_size, hidden_sizes, latent_size_mean, latent_size_var)
        self.noiseless_hits_layer = GenerateNoiselessHitsLayer()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)   
        logvar = torch.clamp(logvar, min=-10, max=10)
        # print("z was ", z[0])
        reconstructed_hits = self.noiseless_hits_layer(z)
        # print("reconstructed_hits was ", reconstructed_hits[0])
        return reconstructed_hits, mu, logvar
    
def compute_differences_batch(set1, set2):
    c2 = torch.sum((set1 - set2) ** 2, dim=(1, 2))
    return torch.mean(c2)
    
def train_encoder(encoder, encoder_optimizer, encoder_scheduler, num_epochs, train_dl, val_dl, device, results_file_encoder='', prev_encoder_path =''):
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
            file.write("Epoch,Train Total,Val Total,Train Recon,Val Recon\n")  # Write the header
    best_encoder_epoch = 1
    best_encoder_loss = float('inf')

    for epoch in range(start_epoch, num_epochs + 1):
        encoder.train()
        total_encoder_loss = 0
        total_kld_loss = 0
        total_recon_loss = 0
        length = 0
        bar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for points, params in bar:
            points = points.float().to(device)
            params = params.float().to(device)

            encoder_optimizer.zero_grad()
            reconstructed_points, mean, logvar = encoder(points)
            input_points = points.view(points.shape[0], 10, 3)
            recon_loss = compute_differences_batch(reconstructed_points, input_points) / points.shape[0]
            kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kld_loss = kld_loss / points.shape[0]
            encoder_loss = recon_loss + KLD_WEIGHT * kld_loss
            encoder_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            encoder_optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            total_encoder_loss += encoder_loss.item()
            length += 1
            
            bar.set_postfix(loss=np.mean(total_encoder_loss / length))

        avg_train_encoder_loss = total_encoder_loss / length
        avg_train_recon_loss = total_recon_loss / length

        encoder.eval()
        with torch.no_grad():
            total_encoder_loss_val = 0
            total_kld_loss_val = 0
            total_recon_loss_val = 0
            length = 0
            for points, params in val_dl:
                points = points.float().to(device)
                params = params.float().to(device)

                reconstructed_points, mean, logvar = encoder(points)
                input_points = points.view(points.shape[0], 10, 3)
                recon_loss = compute_differences_batch(reconstructed_points, input_points) / points.shape[0]
                kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                kld_loss = kld_loss / points.shape[0]
                encoder_loss = recon_loss + KLD_WEIGHT * kld_loss

                total_encoder_loss_val += encoder_loss.item()
                total_kld_loss_val += kld_loss.item()
                total_recon_loss_val += recon_loss.item()
                length += 1

        avg_val_encoder_loss = total_encoder_loss_val / length
        avg_val_recon_loss = total_recon_loss_val / length

        # Write epoch results to file
        with open(results_file_encoder, 'a') as file:
            file.write(f"{epoch},{avg_train_encoder_loss:.7f},{avg_val_encoder_loss:.7f},{avg_train_recon_loss:.7f},{avg_val_recon_loss:.7f}\n")

        # save model
        if epoch >= (3 * ENCODER_EPOCH / 4) and epoch % 5 == 0:
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
            if best_encoder_loss > avg_train_encoder_loss + avg_val_encoder_loss:
                best_encoder_loss = avg_train_encoder_loss + avg_val_encoder_loss
                best_encoder_epoch = epoch

        if encoder_scheduler:
            encoder_scheduler.step()
    with open(results_file_encoder, 'a') as file:  
        file.write(f"using encoder from epoch {best_encoder_epoch}\n") 
    return encoder, encoder_optimizer, best_encoder_epoch, best_encoder_loss

def writeTrainingInfo(encoder_scheduler, decoder_scheduler):
    s = f'''For: {ENCODER_RESULTS_PATH}
# of encoder epochs: {ENCODER_EPOCH}
# of decoder epochs: {DECODER_EPOCH}
learning rate: {LEARNING_RATE}
batch size: {BATCH_SIZE}
encoder milestones: {encoder_scheduler.milestones if encoder_scheduler else "None"}
decoder milestones: {decoder_scheduler.milestones if decoder_scheduler else "None"}\n\n'''
    with open(REPORT_PATH, 'a') as file:
        file.write(s)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = Dataset(path=DATASET_PATH)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset = Dataset(path=VAL_DATASET_PATH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    encoder = VAE_EncoderWithHits(latent_size_var=1)
    encoder = encoder.to(device)

    if torch.cuda.is_available():
        encoder.cuda()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = LEARNING_RATE)

    # encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[20, 60,100], gamma=0.5)
    # encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[100], gamma=0.5)
    encoder_scheduler = None
    decoder_scheduler = None
    
    train_encoder(encoder, encoder_optimizer, encoder_scheduler, ENCODER_EPOCH, train_dataloader, val_dataloader, device, 
                  results_file_encoder=ENCODER_RESULTS_PATH+'encoder_results.csv')
    
    writeTrainingInfo(encoder_scheduler, decoder_scheduler)