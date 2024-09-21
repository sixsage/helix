from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error

TRAIN_PATH = 'tracks_1m_updated_non_gaussian_wider.txt'
with open(TRAIN_PATH, 'r') as file:
    content = file.read()
data_points = content.split('EOT')

data_points = [dp.strip() for dp in data_points if dp.strip()]
data_points = [dp.split('\n') for dp in data_points]
data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
targets = [dp[0] for dp in data_points]
np_targets = np.array(targets)
input_points = [dp[1:] for dp in data_points]
inputs = []
for input in input_points:
    combined = []
    for coordinate in input:
        combined += coordinate
    inputs.append(combined)
inputs = np.array(inputs)

model = XGBRegressor(n_estimators=1000, device="cuda")
model.fit(inputs, np_targets)

VAL_PATH = 'tracks_100k_updated_non_gaussian_wider.txt'
with open(VAL_PATH, 'r') as file:
    content = file.read()
data_points = content.split('EOT')

data_points = [dp.strip() for dp in data_points if dp.strip()]
data_points = [dp.split('\n') for dp in data_points]
data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
targets = [dp[0] for dp in data_points]
test_targets = np.array(targets)
input_points = [dp[1:] for dp in data_points]
inputs = []
for input in input_points:
    combined = []
    for coordinate in input:
        combined += coordinate
    inputs.append(combined)
test_inputs = np.array(inputs)

min_per_column = np.min(test_targets, axis=0)
max_per_column = np.max(test_targets, axis=0)
range_per_column = max_per_column - min_per_column
print("ranges")
print(range_per_column)


predictions = model.predict(test_inputs)
print(test_targets)
print(predictions)
result_mae = np.abs(test_targets - predictions)
result_mae = np.mean(result_mae, axis=0)
print('mae')
print(result_mae)
print('mae / range * 100')
percent_error = (result_mae / range_per_column) * 100
print(percent_error)