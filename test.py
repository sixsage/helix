import torch
import numpy as np
sigma=0.01

# d = torch.tensor([[[ 2.8411e-02, -3.5429e-03,  8.1873e-01],
#          [ 5.8360e-02,  3.2781e-04,  9.5374e-01],
#          [ 8.8888e-02, -3.8311e-03,  1.0917e+00],
#          [ 1.1331e-01, -3.0996e-03,  1.2220e+00],
#          [ 1.4534e-01, -2.8378e-03,  1.3598e+00],
#          [ 1.7443e-01, -5.3164e-03,  1.5002e+00],
#          [ 2.1516e-01, -1.3420e-02,  1.6379e+00],
#          [ 2.3307e-01,  6.3506e-03,  1.7765e+00],
#          [ 2.4726e-01,  3.2443e-03,  1.9130e+00],
#          [ 2.9171e-01,  3.1977e-02,  2.0503e+00]]])

# t = torch.tensor([[[ 0.1900,  0.9800,  0.6900],
#          [ 0.3200,  1.9800,  0.8300],
#          [ 0.3400,  2.9800,  0.9800],
#          [ 0.3100,  3.9800,  1.1200],
#          [ 0.1800,  4.9900,  1.2600],
#          [-0.0100,  6.0100,  1.3900],
#          [-0.3200,  6.9900,  1.5300],
#          [-0.7000,  7.9500,  1.6700],
#          [-1.1600,  8.9200,  1.8100],
#          [-1.7100,  9.8500,  1.9700]]])

# print(torch.square(d - t))
# print(torch.sum(torch.square(d - t)))
# print(torch.norm(d - t, 2, dim=(1, 2)))

# # Example numpy arrays
# d = np.array([[[ 2.8411e-02, -3.5429e-03,  8.1873e-01],
#          [ 5.8360e-02,  3.2781e-04,  9.5374e-01],
#          [ 8.8888e-02, -3.8311e-03,  1.0917e+00],
#          [ 1.1331e-01, -3.0996e-03,  1.2220e+00],
#          [ 1.4534e-01, -2.8378e-03,  1.3598e+00],
#          [ 1.7443e-01, -5.3164e-03,  1.5002e+00],
#          [ 2.1516e-01, -1.3420e-02,  1.6379e+00],
#          [ 2.3307e-01,  6.3506e-03,  1.7765e+00],
#          [ 2.4726e-01,  3.2443e-03,  1.9130e+00],
#          [ 2.9171e-01,  3.1977e-02,  2.0503e+00]]])

# t = np.array([[[ 0.1900,  0.9800,  0.6900],
#          [ 0.3200,  1.9800,  0.8300],
#          [ 0.3400,  2.9800,  0.9800],
#          [ 0.3100,  3.9800,  1.1200],
#          [ 0.1800,  4.9900,  1.2600],
#          [-0.0100,  6.0100,  1.3900],
#          [-0.3200,  6.9900,  1.5300],
#          [-0.7000,  7.9500,  1.6700],
#          [-1.1600,  8.9200,  1.8100],
#          [-1.7100,  9.8500,  1.9700]]])

array1 = np.random.rand(3, 3, 3)  # Replace this with your actual array1
array2 = np.random.rand(3, 3, 3) 

print(array1)
print(array2)
print(1 * (array2 < 0.5))
# # Step 1: Compute pairwise Euclidean distances between the corresponding points
# distances = np.linalg.norm(array1 - array2, axis=-1)  # Shape (100000, 10)
# print("distances")
# print(distances)

# # Step 2: Sum distances for each set of 10 points
# total_distance_per_entry = np.sum(distances, axis=-1)  # Shape (100000,)
# distances_2 = np.sqrt(np.sum(np.square(array1 - array2), axis=2))
# # print(distances_2)

# # Step 3: Sum the total distances over all 100,000 entries
# # total_distance = np.sum(total_distance_per_entry)

# print("Total distance:", total_distance_per_entry)

# a1_tensor = torch.tensor(array1)
# a2_tensor = torch.tensor(array2)
# print("torch")
# print(torch.norm(a1_tensor - a2_tensor, 2, dim=(1, 2)))

# distances_3 = np.linalg.norm(array1 - array2, axis=(1, 2))
# print(distances_3)
# print(np.sum(np.square(d - t)))
# print(np.linalg.norm(d -t))

# print("#####################################")
# print(array1 - array2)
# print(np.square(array1 - array2))
# print(np.sum(np.square(array1 - array2), axis=2))
# print(np.sqrt(np.sum(np.square(array1 - array2), axis=2)))
# print(np.sum(np.sqrt(np.sum(np.square(array1 - array2), axis=2)), axis=1))

# squared_diff = (a1_tensor - a2_tensor) ** 2
# pairwise_distances = torch.sqrt(squared_diff.sum(dim=-1))
# sum_of_distances = pairwise_distances.sum(dim=-1)
# print(sum_of_distances)

# print(np.concatenate((array1, array2), axis=0))