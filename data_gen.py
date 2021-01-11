import gc
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import copy

# DeepMICALib is a helper library for working with Ion Channel Markov Models for data science
import DeepMICALib as dm

from DeepMICALib.generate import *


# Define rate constants as in paper
c = np.array([1, 0.37, 286, 148, 811, 5144, 2395, 37, 1856])

# 5 state model taken from Zhou et al.
transition_matrix = np.array([
    [-(c[1]), 0, 0, 0, c[1]],
    [0, -(c[3]), 0, 0, c[3]],
    [0, 0, -(c[5]), c[5], 0],
    [0, 0, c[6], -(c[6] + c[7]), c[7]],
    [c[2],c[4], 0, c[8], -(c[2]+c[4]+c[8])]
])

print(transition_matrix)


state_dictionary = {
    "Closed 1": 0,
    "Closed 2": 0,
    "Closed 3": 0,
    "Open 1": 1,
    "Open 2": 1
}

def get_noise():
    # Initialise noise object
    noise = Noise()

    # Randomly generate some values for the noise parameters. Empirically decided.

    noise_freq = random.uniform(0.8, 1.2)
    noise_scale_factor = random.uniform(1.0, 1.3)
    noise_amp = random.uniform(0.15, 0.3)

    drift_freq = random.uniform(3, 4)
    drift_amp = random.uniform(0, 0.2)


    noise.add(scaled_f_noise(exponent=noise_freq, scale_factor=noise_scale_factor, base_sd=noise_amp, channels_index=0))
    noise.add(simple_f_noise(exponent=drift_freq, sd=drift_amp))

    return noise

def run_sim(no_channels, file_path):

    master_net = Network(5, trans_matrix=transition_matrix, state_dict=state_dictionary)

    networks = []

    for i in range(no_channels):
        networks.append(copy(master_net))

    log = MultiMarkovLog(networks)
    # Generate 10mins of recording
    log.simulate_discrete(time=1800, sample_rate=1e4)

    # Generate a "noise" object to use for the continuous simulation.
    out_noise = get_noise()

    # Interpolate this at 10kHz
    log.simulate_continuous(1e4, noise=out_noise)
    log.continuous_aggregate[["Channels", "Time", "Noisy Current"]].to_csv(file_path)

    return log

def generate_batch(number_of_channels, number_of_files, folder_name, file_path):
    for i in range(number_of_files):
        true_file_path = f'{file_path}/{folder_name}'
        run_sim(number_of_channels, file_path=f'{true_file_path}/data_{i}')
    return 0

# Run a sample sim
netty = Network(5, trans_matrix=transition_matrix, state_dict=state_dictionary)
log = MultiMarkovLog([netty])
log.simulate_discrete(time=100, sample_rate=1e4)
out_noise = get_noise()
print(log.discrete_history[0])
templog = log.discrete_history[0].groupby(['State']).mean()
print('Mean time per state')
print(templog)
print('Proportion of times per state')
templog = log.discrete_history[0].groupby(['State']).sum()
templog['Time Spent'] = templog['Time Spent']/sum(templog['Time Spent'])
print(templog)



log.simulate_continuous(1e4, noise=out_noise)
log.sample_data_graph(length=100)
plt.show()

datasets = {
    'single_channel': 1,
    'five_channel': 5,
    'ten_channel': 10}
cwd = os.getcwd()

for key, value in datasets.items():
    generate_batch(value, 48, 'training', f'{cwd}/{key}/')
    gc.collect()
    generate_batch(value, 24, 'validation', f'{cwd}/{key}/')
    gc.collect()
    generate_batch(value, 48, 'testing', f'{cwd}/{key}/')

