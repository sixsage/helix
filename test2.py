
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


min_r0=1.0
max_r0=10.0
nlayers=10
sigma=0.01

DATASET_PATH = 'tracks_2k_updated_no_noise_precise.txt'

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

    return res.x[0], res.fun

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

# calculate the track parameters for a set of spacepoints
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

# generate random track parameters and the associated hits
def gen_tracks(n=1):
    
    tracks=[]
    for i in range(n):
        if (i%1000==0):
            print("Track %d/%d" % (i,n))
        d0=np.fabs(np.random.normal(scale=0.01))
        phi=np.random.uniform(low=0,high=2*np.pi)
        pt=np.random.uniform(low=25,high=200)
        dz=np.random.normal(scale=1.0)
        tanl = np.random.normal(scale=0.3)
        params=(d0,phi,pt,dz,tanl)
        xs,ys,zs = make_hits(params)
        tracks.append([params,xs,ys,zs])
    return tracks
        

# scan each track parameter one at a time, make the hits and plot them
# just to verify the tracks look right
def scan():
    for d0 in np.linspace(0, 0.25, 10):
        params = (d0, 0.0, 25.0, 0.0, 0.2)
        xs, ys, zs = make_hits(params)
        plt.plot(xs, ys, "x", label="d0=%1.2f" % d0)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('X-Y Plane: Varying d0')
    plt.legend()
    plt.savefig("scan_d0_xy.png")
    plt.clf()

    for phi0 in np.linspace(0, 2*np.pi, 10):
        params = (0.0, phi0, 25.0, 0.0, 0.2)
        xs, ys, zs = make_hits(params)
        plt.plot(xs, ys, "x", label="phi0=%1.2f" % phi0)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('X-Y Plane: Varying phi0')
    plt.legend()
    plt.savefig("scan_phi0_xy.png")
    plt.clf()

    for pt in np.linspace(25, 200, 10):
        params = (0.0, 0, pt, 0.0, 0.2)
        xs, ys, zs = make_hits(params)
        plt.plot(xs, ys, "x", label="pt=%1.1f" % pt)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('X-Y Plane: Varying pt')
    plt.legend()
    plt.savefig("scan_pt_xy.png")
    plt.clf()

    for dz in np.linspace(-2.5, 2.5, 10):
        params = (0.0, 0, 25, dz, 0.2)
        xs, ys, zs = make_hits(params)
        plt.plot(xs, zs, "x", label="dz=%1.1f" % dz)
    plt.xlabel('X-coordinate')
    plt.ylabel('Z-coordinate')
    plt.title('X-Z Plane: Varying dz')
    plt.legend()
    plt.savefig("scan_dz_xz.png")
    plt.clf()

    for tanl in np.linspace(-1, 1, 10):
        params = (0.0, 0, 25, 0, tanl)
        xs, ys, zs = make_hits(params)
        plt.plot(xs, zs, "x", label="tanl=%1.1f" % tanl)
    plt.xlabel('X-coordinate')
    plt.ylabel('Z-coordinate')
    plt.title('X-Z Plane: Varying tanl')
    plt.legend()
    plt.savefig("scan_tanl_xz.png")
    plt.clf()
    
def parse_data_ls(filename):
    f = open(filename)
    data = f.readlines()
    tgts = []
    datapoints = []
    features = []
    newdata = False
    for line in data:
        line = line.strip()
        vals = line.split(",")      
        if len(vals) == 5:
            #print(vals)
            tgts.append(vals)

    tgts = np.array(tgts).astype(float)
    return tgts


def view_trajectories(X_test):
    # Reshape X_test to (20000, 10, 3) if the original shape is (10, 3)
    X_test_reshaped = X_test.reshape(-1, 10, 3)
    
    for i in range(1):
        #actual_params = helix_targets[i]
        
        #predicted_params = new_preds[i]
        
        print(X_test_reshaped.shape)
        
        xs_actual = X_test_reshaped[i, :, 0]
        ys_actual = X_test_reshaped[i, :, 1]
        zs_actual = X_test_reshaped[i, :, 2]
        
            
        
        # Optionally plot the hits for visual comparison
        plt.figure(figsize=(10, 5))
        

        # XY Plane
        plt.subplot(1, 2, 1)
        plt.plot(xs_actual, ys_actual, "x", color='red', label='Actual Hits')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Track {i} in XY Plane')
        plt.legend()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.tight_layout()
        plt.show()
        

        # XZ Plane
        plt.subplot(1, 2, 2)
        plt.plot(xs_actual, zs_actual, "x", color='red', label='Actual Hits')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title(f'Track {i} in XZ Plane')
        plt.legend()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.tight_layout()
        plt.show()

        #plt.tight_layout()
        #plt.savefig(f"plot_{i}.pdf")
        plt.clf()

def view_trajectories_compare(X_test, X_test2):
    # Reshape X_test to (20000, 10, 3) if the original shape is (10, 3)
    X_test_reshaped = X_test.reshape(-1, 10, 3)
    X_test2_reshaped = X_test2.reshape(-1, 10, 3)
    
    for i in range(1):
        #actual_params = helix_targets[i]
        
        #predicted_params = new_preds[i]
        
        #print(X_test_reshaped.shape)
        
        xs_actual = X_test_reshaped[i, :, 0]
        ys_actual = X_test_reshaped[i, :, 1]
        zs_actual = X_test_reshaped[i, :, 2]

        xs_actual2 = X_test2_reshaped[i, :, 0]
        ys_actual2 = X_test2_reshaped[i, :, 1]
        zs_actual2 = X_test2_reshaped[i, :, 2]
        
            
        
        # Optionally plot the hits for visual comparison
        plt.figure(figsize=(10, 5))
        

        # XY Plane
        plt.subplot(1, 2, 1)
        plt.plot(xs_actual, ys_actual, "x", color='red', label='Actual Hits')
        plt.plot(xs_actual2, ys_actual2, "x", color='blue', label='Actual Hits 2')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Track {i} in XY Plane')
        plt.legend()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.tight_layout()
        plt.show()
        

        # XZ Plane
        plt.subplot(1, 2, 2)
        plt.plot(xs_actual, zs_actual, "x", color='red', label='Actual Hits')
        plt.plot(xs_actual2, zs_actual2, "x", color='blue', label='Actual Hits 2')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title(f'Track {i} in XZ Plane')
        plt.legend()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.tight_layout()
        plt.show()

        #plt.tight_layout()
        #plt.savefig(f"plot_{i}.pdf")
        plt.clf()
        
def compute_differences(set1, set2):
    """
    Computes the sum of squared differences between two sets of 10 points.

    Parameters:
    - set1: An array of shape (10, 3) representing the first set of points.
    - set2: An array of shape (10, 3) representing the second set of points.

    Returns:
    - c2: The sum of squared differences between the corresponding points in set1 and set2.
    """
    if set1.shape != (10, 3) or set2.shape != (10, 3):
        raise ValueError("Both input sets must have shape (10, 3).")

    # Compute the sum of squared differences
    #print((set1 - set2) ** 2)
    c2 = np.sum((set1 - set2) ** 2)
    return c2
        
        
        
        
import time

def make_hits_noiseless_orig(params):
    xs=[]
    ys=[]
    zs =[]
    total_dr = 0
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0, min_dr = find_phi(r0*r0,*params)
        #print("min dr was ", min_dr)
        #print(" r0 = ",r0, " phi0 = ",phi0)
        #fphi0= fast_find_phi(r0*r0,*params)
        #print(" fr0 = ",r0, " fphi0 = ",phi0)
        x0,y0,z0 = track(phi0,*params)
        shape_param = 10  # Higher values reduce skew
        
        xs.append(x0)
        ys.append(y0)
        zs.append(z0)
        
        total_dr += min_dr

    out = np.column_stack((xs, ys, zs))

    return out, total_dr / nlayers


    
    
    
# Load and preprocess your data
f = open(DATASET_PATH)
data = f.readlines()

tgts = []
datapoints = []
features = []
newdata = False
for line in data:
    line = line.strip()
    vals = line.split(",")
    if newdata and len(vals) >= 3 and len(features) == 10:
        datapoints.append(features)
        newdata = False
        features = []
    
    if len(vals) == 5:
        tgts.append(vals)
    if len(vals) == 3:
        features.append(vals)
    else:
        newdata = True

datapoints.append(features)

datapoints = np.array(datapoints).astype(float)
flattened_data = datapoints.reshape(datapoints.shape[0], 30)
tgts = np.array(tgts).astype(float)
new_tgts = tgts

# Shuffle and prepare training data
combined_data = list(zip(flattened_data, new_tgts))

shuffled_flattened_data, shuffled_new_tgts = zip(*combined_data)
shuffled_flattened_data = np.array(shuffled_flattened_data)
shuffled_new_tgts = np.array(shuffled_new_tgts)

X_train = shuffled_flattened_data
y_train = shuffled_new_tgts



ls_predictions = parse_data_ls(DATASET_PATH)

#ls_predictions = ls_predictions[0:100]
#X_train = X_train[0:100]
#y_train = y_train[0:100]

true_X = []

for params in y_train:
    hits, _ = make_hits_noiseless_orig(params)
    true_X.append(hits)

true_X = np.array(true_X)




import torch
import numpy as np
import time
from torch.optim import LBFGS
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR

# Wrap angle to be within [0, 2π]
def wrap_angle_to_2pi(angle):
    return angle % (2 * torch.pi)

# Compute the 3D track position based on helical parameters
def compute_3d_track(phi, d0, phi0, pt, dz, tanl):
    alpha = 1 / 2  # 1/cB
    q = 1
    kappa = q / pt
    rho = alpha / kappa
    x = d0 * torch.cos(phi0) + rho * (torch.cos(phi0) - torch.cos(phi0 + phi))
    y = d0 * torch.sin(phi0) + rho * (torch.sin(phi0) - torch.sin(phi0 + phi))
    z = dz - rho * tanl * phi
    return x, y, z

def scale_phi(phi_raw):
    return 2 * torch.pi * torch.sigmoid(phi_raw)


def optimize_phi_differentiable_lbfgs(phi_init, target_radius_squared, d0, phi0, pt, dz, tanl, n_iters=100, lr=0.01, device='cpu'):

    # Scale sigmoid output to [0, 2π]
    #phi_raw = phi_init.clone().detach().requires_grad_(True).to(device)
    phi_raw = torch.nn.Parameter(phi_init.clone().to(device))

    optimizer = torch.optim.LBFGS([phi_raw], lr=lr, max_iter=n_iters)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch < 5 else 1.0)


    #print("Initial raw phi is ", phi_raw)
    #print("initial phi is ", scale_phi(phi_raw))

    #print(f"Initial phi_init: requires_grad={phi_init.requires_grad}, grad_fn={phi_init.grad_fn}")


    def closure():
        optimizer.zero_grad()

        # Scale raw phi to [0, 2π]
        phi = scale_phi(phi_raw)

        # Compute the track position
        x, y, z = compute_3d_track(phi, d0, phi0, pt, dz, tanl)       

        radius_squared = x * x + y * y

        # Compute the loss
        loss = (radius_squared - target_radius_squared).pow(2)

        # Check stopping condition
        if loss.item() < 0.001:
            #print("Early stopping: Loss is below threshold")
            raise StopIteration  # Stop optimization by raising an exception


        loss.backward(retain_graph=True)

        
        return loss

    try:
        optimizer.step(closure)
    except StopIteration:
        pass  # Gracefully catch the exception and exit optimization


    phi_optimized = scale_phi(phi_raw)


    x, y, z = compute_3d_track(phi_optimized, d0, phi0, pt, dz, tanl)

    radius_squared = x * x + y * y



    final_loss = torch.abs(radius_squared - target_radius_squared).detach()

    return phi_optimized, final_loss



def sigmoid_inverse(x):
    return torch.log(x / (1 - x))

def scale_phi_inverse(phi_scaled):
    # Map back to (0, 1)
    x = phi_scaled / (2 * torch.pi)
    # Apply sigmoid inverse
    return torch.log(x / (1 - x))


# Generate noiseless hit points in 3D using optimized phi values
def generate_noiseless_hits_sequential(params, min_radius=1.0, max_radius=10.0, num_layers=10, n_iters=1000, lr=0.01, device='cpu'):
    #d0, phi0, pt, dz, tanl = [torch.tensor(param, requires_grad=True, device=device) for param in params]
    
    #print(params.requires_grad)
    
    d0, phi0, pt, dz, tanl = params
    
    #return params * 5
    
    #return sigmoid_inverse(params)

    xs, ys, zs = [], [], []
    total_distance = 0
    optimized_phi_values = []

    # Start with an initial phi value for the first radius
    #phi_init = torch.tensor(0.02, device=device, requires_grad=True)

    #phi_init = sigmoid_inverse(phi_init)

    phi_init = torch.tensor(0.2 / (2 * torch.pi), device=device, requires_grad=True)
    phi_init = sigmoid_inverse(phi_init)

    #print("Initial phi_init:", phi_init)
    #print("phi_init requires_grad:", phi_init.requires_grad)
    #print("phi_init grad_fn:", phi_init.grad_fn)

    target_radii = [torch.tensor(r, device=device, requires_grad=True) for r in torch.linspace(min_radius, max_radius, num_layers)]


    #for target_radius in torch.linspace(min_radius, max_radius, num_layers, device=device):
    for target_radius in target_radii:
        # Optimize phi for the current radius
        phi, distance_residual = optimize_phi_differentiable_lbfgs(
            phi_init, target_radius**2, d0, phi0, pt, dz, tanl, n_iters, lr, device
        )
        

        optimized_phi_values.append(phi)

        print(phi)
        # Compute 3D position with the optimized phi
        x, y, z = compute_3d_track(phi, d0, phi0, pt, dz, tanl)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        total_distance += distance_residual

        #print("best phi before scaling ", phi)

        # Use the current optimized phi as the starting point for the next radius
        #phi_init = phi.clone().detach().requires_grad_(True)
        #phi_init = scale_phi_inverse(phi)
        #phi_init = scale_phi_inverse(phi).detach().requires_grad_(True)
        phi_init = scale_phi_inverse(phi)

        #print("phi before being input into optimization ", phi_init)

        #phi_init = sigmoid_inverse(phi_init)

        #print("best phi was ", sigmoid_inverse(phi_init))
        #break
        

    # Stack results into a tensor
    hit_points = torch.stack((torch.stack(xs), torch.stack(ys), torch.stack(zs)), dim=1)

    #print("optimized phi gradient info ", optimized_phi_values[0].grad_fn)
    #print("optimized phi gradient info ", optimized_phi_values[1].grad_fn)


    #return hit_points, total_distance / num_layers, torch.stack(optimized_phi_values)
    return hit_points




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ls_predictions = parse_data_ls(DATASET_PATH)

ls_predictions_diff = torch.tensor(ls_predictions, requires_grad=True)

test_point = torch.tensor(ls_predictions_diff[42], requires_grad=True)

#hits, avg_dr, phi_vals = generate_noiseless_hits_sequential(ls_predictions_diff[40], device=device)
print('before out')
out = generate_noiseless_hits_sequential(test_point, device=device)



# Define a dummy loss: Mean squared distance to a target output
#target_output = torch.zeros_like(out)  # Example target: zero positions
#loss = torch.mean((out - target_output) ** 2)  # Only include the MSE of output

true_X_torch = torch.tensor(true_X, requires_grad=True, device=out.device)
print(true_X_torch.shape)

loss = torch.mean((torch.tensor(true_X_torch[42], requires_grad=True, device=out.device) - out) ** 2)

print("loss was ", loss)

# Backpropagate to test gradients
loss.backward()

# Print gradients for each parameter
print("Gradients of params:", test_point.grad)


print(out)
print(true_X[42])


test_train = X_train[0]

test_train = torch.tensor(test_train, requires_grad=True, device=out.device).float()


import torch
import torch.nn as nn
import torch.optim as optim

# Define the encoder network
class Encoder(nn.Module):
    def __init__(self, input_size=30, latent_size=5):
        super(Encoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU()
        )
        self.fc_latent = nn.Linear(100, latent_size)

    def forward(self, x):
        x = self.fc_layers(x)
        x = self.fc_latent(x)
        return x

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder().to(device)

# Define optimizer and loss
optimizer = optim.Adam(encoder.parameters(), lr=0.001)
mse_loss = nn.MSELoss()


# Pretraining: Define the target helical parameters for pretraining
pretrain_target = torch.tensor(y_train[0], dtype=torch.float32, device=device)

# Define the pretraining optimizer and loss function
pretrain_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
pretrain_loss_fn = nn.MSELoss()

# Pretraining loop
pretrain_epochs = 100  # Number of pretraining epochs
for epoch in range(pretrain_epochs):
    pretrain_optimizer.zero_grad()

    # Encode the input to predict helical parameters
    predicted_params = encoder(test_train)

    # Compute the pretraining loss (difference between predicted and true helical parameters)
    pretrain_loss = pretrain_loss_fn(predicted_params, pretrain_target)

    # Backpropagation
    pretrain_loss.backward()
    pretrain_optimizer.step()

    # Print pretraining loss for monitoring
    if (epoch + 1) % 10 == 0:
        print(f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}], Loss: {pretrain_loss.item():.4f}")

print("Pretraining completed. Proceeding to main training...")

# Main Training Loop (unchanged from before)
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Encode the input to latent space
    latent_space = encoder(test_train)

    # Decode latent space using the function
    reconstructed_points = generate_noiseless_hits_sequential(latent_space.squeeze(), device=device)

    # Reshape input to (10, 3)
    input_points = test_train.view(10, 3)

    # Compute the loss
    print(latent_space)
    print(reconstructed_points)
    print(input_points)
    loss = mse_loss(reconstructed_points, input_points)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Print loss for monitoring
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
save_path = "temp_save\\encoder_simple.pth"
torch.save(encoder.state_dict(), save_path)
print(f"Encoder model saved to {save_path}")
