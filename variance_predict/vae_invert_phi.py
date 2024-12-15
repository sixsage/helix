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
ENCODER_EPOCH = 60
DECODER_EPOCH = 200
REPORT_PATH = 'experiments.txt'


# mixture
# DATASET_PATH = 'non_gaussian_tracks\\tracks_1M_gaussian_mixturev2.txt'
# VAL_DATASET_PATH = 'non_gaussian_tracks\\tracks_100k_gaussian_mixturev2.txt'
# ENCODER_RESULTS_PATH = 'variance_predict\\mixture\\results3\\'
# DECODER_RESULTS_PATH = 'variance_predict\\mixture\\results3\\'

DATASET_PATH = 'tracks_1M_updated.txt'
VAL_DATASET_PATH = 'tracks_100k_updated.txt'
ENCODER_RESULTS_PATH = 'variance_predict\\gaussian\\vae_results1\\'
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
    
class MSEWithVarianceConstraint(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
    
    def forward(self, mse_value, variance):
        mse_term = mse_value / (2 * variance)  
        variance_term = 0.5 * torch.log(variance)
        return torch.mean(mse_term + variance_term)
    
def find_phi_inverse(target_radius_squared, d0, phi0, pt, dz, tanl, eps=1e-6):
    alpha = 1 / 2  # 1/cB
    q = 1
    kappa = q / pt
    rho = alpha / kappa
    # radius_sqaured == target_radius_squared (try to)
    # radius_squared == x^2 + y ^ 2
    # (a − rho * cos(phi0 + phi)) ^ 2 + (c − rho * sin(phi0+phi)) ^ 2 == target_radius_squared
    # a^2 + c^2 - 2*a*rho*cos(phi0 + phi) - 2*c*rho*sin(phi0+phi) + rho^2 == target_radius_sqaured
    # a^2 + c^2 + rho^2 - target_radius_squred = 2*a*rho*cos(phi0 + phi) + 2*c*rho*sin(phi0+phi)
    # (a^2 + c^2 + rho^2 - target_radius_squred) / (2 * rho) = a * cos(phi0 + phi) + c * sin(phi0 + phi)
    # arctan(c / a) = phi0, hypotenuse = sqrt(a ^ 2 + c ^ 2)
    # a * cos(phi0 + phi) + c * sin(phi0 + phi) = hypotenuse * cos(phi0 + phi - phi0)
    # (a^2 + c^2 + rho^2 - target_radius_squred) / (2 * rho) = hypotenuse * cos(phi0 + phi - phi0)
    # arccos( (a^2 + c^2 + rho^2 - target_radius_squred) / ((2 * rho) * hypotenuse) ) = phi0 + phi - phi0
    # arccos( (a^2 + c^2 + rho^2 - target_radius_squred) / ((2 * rho) * hypotenuse) ) = phi

    # a = calcualted_term_x, c = calculated_term_y

    calculated_term_x = d0 * torch.cos(phi0) + rho * torch.cos(phi0) 
    calculated_term_y = d0 * torch.sin(phi0) + rho * torch.sin(phi0)

    hypotenuse = torch.sqrt(calculated_term_x ** 2 + calculated_term_y ** 2)

    arccos_input = (calculated_term_x ** 2 + calculated_term_y ** 2 + rho ** 2 - target_radius_squared) / (2 * rho * hypotenuse)
    clamped = torch.clamp(arccos_input, min=-1 + eps, max=1-eps) # to avoid nan
    arccos_term = torch.acos(clamped)
    phi = arccos_term 

    # print('----------')
    # print(arccos_input)
    # print(arccos_term)

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
def generate_noiseless_hits_sequential(params, min_radius=1.0, max_radius=10.0, num_layers=10, n_iters=1000, lr=0.01, device='cpu'):
    #d0, phi0, pt, dz, tanl = [torch.tensor(param, requires_grad=True, device=device) for param in params]
    
    #print(params.requires_grad)
    d0, phi0, pt, dz, tanl = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
    
    #return params * 5
    
    #return sigmoid_inverse(params)

    xs, ys, zs = [], [], []
    total_distance = 0

    # Start with an initial phi value for the first radius
    #phi_init = torch.tensor(0.02, device=device, requires_grad=True)

    #phi_init = sigmoid_inverse(phi_init)

    # phi_init = torch.tensor(0.2 / (2 * torch.pi), device=device, requires_grad=True)
    # phi_init = sigmoid_inverse(phi_init)

    #print("Initial phi_init:", phi_init)
    #print("phi_init requires_grad:", phi_init.requires_grad)
    #print("phi_init grad_fn:", phi_init.grad_fn)

    target_radii = [torch.tensor(r, device=device, requires_grad=True) for r in torch.linspace(min_radius, max_radius, num_layers)]


    #for target_radius in torch.linspace(min_radius, max_radius, num_layers, device=device):
    for target_radius in target_radii:
        # Optimize phi for the current radius
        phi = find_phi_inverse(
            target_radius**2, d0, phi0, pt, dz, tanl
        )

        # print(phi)
        # Compute 3D position with the optimized phi
        x, y, z = compute_3d_track(phi, d0, phi0, pt, dz, tanl)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        # total_distance += distance_residual

        # phi2 = find_phi_inverse(target_radius**2, d0, phi0, pt, dz, tanl)
        # print(phi2)
        # print('----------')
        # x, y, z = compute_3d_track(phi2, d0, phi0, pt, dz, tanl)
        # xs.append(x)
        # ys.append(y)
        # zs.append(z)

        #print("best phi before scaling ", phi)

        # Use the current optimized phi as the starting point for the next radius
        #phi_init = phi.clone().detach().requires_grad_(True)
        #phi_init = scale_phi_inverse(phi)
        #phi_init = scale_phi_inverse(phi).detach().requires_grad_(True)
        # phi_init = scale_phi_inverse(phi)

        #print("phi before being input into optimization ", phi_init)

        #phi_init = sigmoid_inverse(phi_init)

        #print("best phi was ", sigmoid_inverse(phi_init))
        #break
        

    # Stack results into a tensor
    hit_points = torch.stack((torch.stack(xs, dim=1), torch.stack(ys, dim=1), torch.stack(zs, dim=1)), dim=2)

    # 10, 3, 4 to 4, 10, 3

    #print("optimized phi gradient info ", optimized_phi_values[0].grad_fn)
    #print("optimized phi gradient info ", optimized_phi_values[1].grad_fn)


    #return hit_points, total_distance / num_layers, torch.stack(optimized_phi_values)
    return hit_points
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = nn.Linear(30, 32)
        # self.layer2 = nn.Linear(32, 64)
        # self.layer3 = nn.Linear(64, 5)
        self.layer1_mean = nn.Linear(30, 200)
        self.layer2_mean = nn.Linear(200, 400)
        self.layer3_mean = nn.Linear(400, 800)
        self.layer4_mean = nn.Linear(800, 800)
        self.layer5_mean = nn.Linear(800, 800)
        self.layer6_mean = nn.Linear(800, 400)
        self.layer7_mean = nn.Linear(400, 200)
        self.output_layer_mean = nn.Linear(200, 5)

        # self.layer1_var = nn.Linear(30, 200)
        # self.layer2_var = nn.Linear(200, 400)
        # self.layer3_var = nn.Linear(400, 800)
        # self.layer4_var = nn.Linear(800, 800)
        # self.layer5_var = nn.Linear(800, 400)
        # self.layer6_var = nn.Linear(400, 200)
        # self.output_layer_var = nn.Linear(200, 5)


    def forward(self, input):
        x = F.leaky_relu(self.layer1_mean(input))
        x = F.leaky_relu(self.layer2_mean(x))
        x = F.leaky_relu(self.layer3_mean(x))
        x = F.leaky_relu(self.layer4_mean(x))
        x = F.leaky_relu(self.layer5_mean(x))
        x = F.leaky_relu(self.layer6_mean(x))
        x = F.leaky_relu(self.layer7_mean(x))
        mean = self.output_layer_mean(x)

        # x = F.relu(self.layer1_var(input))
        # x = F.relu(self.layer2_var(x))
        # x = F.relu(self.layer3_var(x))
        # x = F.relu(self.layer4_var(x))
        # x = F.relu(self.layer5_var(x))
        # x = F.relu(self.layer6_var(x))
        # var = F.relu(self.output_layer_var(x))

        return mean

    
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
            file.write("Epoch,Train Loss,Val Loss,MSE Train,MSE Val\n")  # Write the header
    best_encoder_epoch = 1
    best_encoder_loss = float('inf')

    mse_loss_mean_reduction = nn.MSELoss()

    for epoch in range(start_epoch, num_epochs + 1):
        encoder.train()
        train_losses_encoder = []

        bar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for points, params in bar:
            points = points.float().to(device)
            params = params.float().to(device)

            encoder_optimizer.zero_grad()
            mean = encoder(points)
            reconstructed_points = generate_noiseless_hits_sequential(mean, device=device)
            input_points = points.view(points.shape[0], 10, 3)
            encoder_loss = mse_loss_mean_reduction(reconstructed_points, input_points)
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

                mean = encoder(points)
                reconstructed_points = generate_noiseless_hits_sequential(mean, device=device)
                input_points = points.view(points.shape[0], 10, 3)
                val_loss = mse_loss_mean_reduction(reconstructed_points, input_points)

                val_losses_encoder.append(val_loss.item())

        avg_val_loss_encoder = np.mean(val_losses_encoder)

        # Write epoch results to file
        with open(results_file_encoder, 'a') as file:
            file.write(f"{epoch},{avg_train_loss_encoder:.7f},{avg_val_loss_encoder:.7f}\n")

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
            if best_encoder_loss > avg_train_loss_encoder + avg_val_loss_encoder:
                best_encoder_loss = avg_train_loss_encoder + avg_val_loss_encoder
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

    encoder = Encoder()
    encoder = encoder.to(device)
    # decoder = Decoder()
    # decoder = decoder.to(device)

    if torch.cuda.is_available():
        encoder.cuda()
        # decoder.cuda()
    # encoder_criterion = MSEWithVarianceConstraint()
    encoder_criterion = MSEWithVarianceConstraint(device=device)
    decoder_criterion = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = LEARNING_RATE)
    # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = LEARNING_RATE)
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[20, 60,100], gamma=0.5)
    # decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[20, 60,100], gamma=0.5)
    # encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[100], gamma=0.5)
    # decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[100], gamma=0.5)
    # encoder_scheduler = None
    decoder_scheduler = None
    
    train_encoder(encoder, encoder_optimizer, encoder_scheduler, ENCODER_EPOCH, train_dataloader, val_dataloader, device, 
                  results_file_encoder=ENCODER_RESULTS_PATH+'encoder_results.csv')
    
    writeTrainingInfo(encoder_scheduler, decoder_scheduler)