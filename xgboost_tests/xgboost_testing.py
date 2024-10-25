from xgboost import XGBRegressor
import numpy as np

TRAIN_PATH = 'tracks_1m_updated_asymmetric_higher.txt'
HELIX_VAL_PATH = 'tracks_100k_updated_asymmetric_higher_test.txt'
NON_HELIX_VAL_PATH = 'sintracks_100k_updated_asymmetric_higher_test.txt'

with open(TRAIN_PATH, 'r') as file:
    content = file.read()
    data_points = content.split('EOT')

    data_points = [dp.strip() for dp in data_points if dp.strip()]
    data_points = [dp.split('\n') for dp in data_points]
    data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
    targets = [dp[0] for dp in data_points]
    PARAMETERS_TRAINING = np.array(targets)
    input_points = [dp[1:] for dp in data_points]
    inputs = []
    for input in input_points:
        combined = []
        for coordinate in input:
            combined += coordinate
        inputs.append(combined)
    POINTS_TRAINING = np.array(inputs)

with open(HELIX_VAL_PATH, 'r') as file:
    content = file.read()
    data_points = content.split('EOT')

    data_points = [dp.strip() for dp in data_points if dp.strip()]
    data_points = [dp.split('\n') for dp in data_points]
    data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
    targets = [dp[0] for dp in data_points]
    PARAMETERS_VALIDATION_HELIX = np.array(targets)
    input_points = [dp[1:] for dp in data_points]
    inputs = []
    for input in input_points:
        combined = []
        for coordinate in input:
            combined += coordinate
        inputs.append(combined)
    POINTS_VALIDATION_HELIX = np.array(inputs)

with open(NON_HELIX_VAL_PATH, 'r') as file:
    content = file.read()
    data_points = content.split('EOT')

    data_points = [dp.strip() for dp in data_points if dp.strip()]
    data_points = [dp.split('\n') for dp in data_points]
    data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
    targets = [dp[0] for dp in data_points]
    PARAMETERS_VALIDATION_NON_HELIX = np.array(targets)
    input_points = [dp[1:] for dp in data_points]
    inputs = []
    for input in input_points:
        combined = []
        for coordinate in input:
            combined += coordinate
        inputs.append(combined)
    POINTS_VALIDATION_NON_HELIX = np.array(inputs)
print(POINTS_VALIDATION_HELIX.shape)
print(PARAMETERS_VALIDATION_HELIX.shape)

def train():
    encoder = XGBRegressor(n_estimators=1500, device="cuda")
    encoder.fit(POINTS_TRAINING, PARAMETERS_TRAINING) # train "encoder"
    # make encoder predict parameters
    predicted_parameters = encoder.predict(POINTS_TRAINING)
    decoder = XGBRegressor(n_estimators=1500, device="cuda")
    decoder.fit(predicted_parameters, PARAMETERS_TRAINING)

    return encoder, decoder

def calc_parameter_error(encoder):
    min_per_column = np.min(PARAMETERS_VALIDATION_HELIX, axis=0)
    max_per_column = np.max(PARAMETERS_VALIDATION_HELIX, axis=0)
    range_per_column = max_per_column - min_per_column
    print("ranges")
    print(range_per_column)
    predictions = encoder.predict(POINTS_VALIDATION_HELIX)
    print(POINTS_VALIDATION_HELIX)
    print(predictions)
    result_mae = np.abs(POINTS_VALIDATION_HELIX - predictions)
    result_mae = np.mean(result_mae, axis=0)
    print('mae')
    print(result_mae)
    print('mae / range * 100')
    percent_error = (result_mae / range_per_column) * 100
    print(percent_error)

def calc_classification(encoder, decoder, threshold):
    predicted_points_helical = decoder(encoder(POINTS_VALIDATION_HELIX)) # this will be 2d
    reshaped_val_helix_points = POINTS_VALIDATION_HELIX.reshape((POINTS_VALIDATION_HELIX.shape[0], 10, 3))
    reshaped_predicted_points_helical = predicted_points_helical.reshape((POINTS_VALIDATION_HELIX.shape[0], 10, 3))
    distances_helical = np.linalg.norm(reshaped_val_helix_points - reshaped_predicted_points_helical, axis=-1)
    total_distance_per_entry_helical = np.sum(distances_helical, axis=-1)

    predicted_points_helical = decoder(encoder(POINTS_VALIDATION_NON_HELIX)) # this will be 2d
    reshaped_val_non_helix_points = POINTS_VALIDATION_HELIX.reshape((POINTS_VALIDATION_NON_HELIX.shape[0], 10, 3))
    reshaped_predicted_points_non_helical = reshaped_val_non_helix_points.reshape((POINTS_VALIDATION_NON_HELIX.shape[0], 10, 3))
    distances_helical = np.linalg.norm(reshaped_val_non_helix_points - reshaped_predicted_points_non_helical, axis=-1)
    total_distance_per_entry_non_helical = np.sum(distances_helical, axis=-1)

    combined_distances = np.concatenate((total_distance_per_entry_helical, total_distance_per_entry_non_helical), axis=0)

    target = np.concatenate((np.ones(total_distance_per_entry_helical.shape), np.zeros(total_distance_per_entry_non_helical.shape)))
    predictions = 1 * (combined_distances < threshold)
    print(target)
    print(predictions)

    print(np.mean(target == predictions))


    