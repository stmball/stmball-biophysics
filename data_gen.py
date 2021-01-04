import numpy as np
import matplotlib.pyplot as plt
import random
from copy import copy

# DeepMICALib is a helper library for working with Ion Channel Markov Models for data science
import DeepMICALib as dm

from DeepMICALib.generate import *


# Define rate constants as in paper
constants = np.array([-7.5617, -8.3559, -4.9268, -6.4825, -6.9486, -4.0923, -7.9836, -8.2472])
c = np.exp(constants)
c *= 10000
# 17 state model taken from Zhou et al.
transition_matrix = np.array([
    [-(4*c[0]), 4*c[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [c[1],-(c[1]+3*c[0]+c[2]),3*c[0], 0, 0, c[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2*c[1], -(2*c[1] + 2*c[0] + 2*c[2]), 2*c[0], 0, 0, 2*c[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3*c[1], -(3*c[1] + c[0] + 3*c[2]), c[0], 0, 0, 3*c[2], 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4*c[1], -(4*c[1] + 4*c[2]), 0, 0, 0, 4*c[2], 0, 0, 0, 0, 0, 0, 0, 0],
    [0, c[3], 0, 0, 0, -(c[3] + 3*c[0]), 3*c[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, c[3], 0, 0, 2*c[1], -(c[3] + 2*c[1] + 2*c[0] + c[2]), 2*c[0], 0, c[2], 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, c[3], 0, 0, 3*c[1], -(c[3] + 3*c[1] + c[0] + 2*c[2]), c[0], 0, 2*c[2], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, c[3], 0, 0, 4*c[1], -(c[3] + 4*c[1] + 3*c[2]), 0, 0, 3*c[2], 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2*c[3], 0, 0, -(2*c[3] + 2*c[0]), 2*c[0], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2*c[3], 0, 3*c[1], -(2*c[3] + 3*c[1] + c[0] + c[2]), c[0], c[2], 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2*c[3], 0, 4*c[1], -(2*c[3] + 4*c[1] + 2*c[2]), 0, 2*c[2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*c[3], 0, -(3*c[3] + c[0]), c[0], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*c[3], 4*c[1], -(3*c[3] + 4*c[1] + c[2]), c[2], 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4*c[3], -(4*c[3] + c[4]), c[4], 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c[5], -(c[5] + c[7]), c[7]],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c[6], -(c[6])]
])


state_dictionary = {
    "Closed 1": 0,
    "Closed 2": 0,
    "Closed 3": 0,
    "Closed 4": 0,
    "Closed 5": 0,
    "Closed 6": 0,
    "Closed 7": 0,
    "Closed 8": 0,
    "Closed 9": 0,
    "Closed 10": 0,
    "Closed 11": 0,
    "Closed 12": 0,
    "Closed 13": 0,
    "Closed 14": 0,
    "Closed 15": 0,
    "Open 1": 1,
    "Open 2": 1
}

def get_noise():
    # Initialise noise object
    noise = Noise()

    # Randomly generate some values for the noise parameters. Empirically decided.

    noise_freq = random.uniform(0.9, 1.1)
    noise_scale_factor = random.uniform(1.0, 1.3)
    noise_amp = random.uniform(0.25, 0.4)

    drift_freq = random.uniform(5, 10)
    drift_amp = random.uniform(0, 0.5)


    noise.add(scaled_f_noise(noise_freq, scale_factor=noise_scale_factor, base_sd=noise_amp))
    noise.add(simple_f_noise(drift_freq, sd=drift_amp))

    return noise


master_net = Network(17, trans_matrix=transition_matrix, state_dict=state_dictionary)

networks = []

for i in range(100):
    networks.append(copy(master_net))

log = MultiMarkovLog(networks)
# Generate 10mins of recording
log.simulate_discrete(time=10, sample_rate=1e4)

# Generate a "noise" object to use for the continuous simulation.
out_noise = get_noise()

# Interpolate this at 10kHz
log.simulate_continuous(1e4, noise=out_noise)

log.sample_data_graph(length=1)
plt.show()
