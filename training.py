# Helper function from the DeepMICA Library
import DeepMICALib as dm
from DeepMICALib.generate import MarkovLog, Network
from DeepMICA.preprocessing import PPData, PPFromFile

# Other imports
import os
import sys
import csv
import tensorflow as tf

from tqdm import tqdm
from functools import lru_cache

# Constants
BATCH_SIZE = 100
WINDOW_SIZE = 1000
MAX_CACHE = 24

HARD_EPOCH_LIMIT = 1000
PLT_LEN_SEC = 1
PLT_LEN_SAM = 10000
SAMPLE_RATE = 10000

# Check if we are using the GPU. If not, training will take much longer!
if tf.test.is_gpu_available():
    print("GPU(s) available, using them.")
else:
    print("No compatible GPU found - are you sure you want to continue? Press any key to do so.")
    input()

# Construct a generator for getting the datasets, so we don't have to read them all from memory.
class PPGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_paths, batch_size, max_states, state_dict, window_size):

        self.batch_size = batch_size
        self.window_size = window_size
        self.max_states = max_states
        self._mapping = {}

        count = 0

        # Read all the files once to determine the datasize.
        print("Determining dataset size")
        for idx, file_path in enumerate(tqdm(file_paths)):
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                size = (len(list(reader)) - 1) // (self.window_size * self.batch_size)

                # Add to the mapping
                self._mapping[(count, count + size)] = file_path
                count += size
        self.no_lines = count

        def _find_file_path(self, idx):
            # Iterate through the mapping to find the file with the requested index
            for my_range, file_path in self._mapping.items():
                start, end = my_range[0], my_range[1]
                if start <= idx and idx < end:
                    in_file_idx = idx - start
                    return (in_file_idx, file_path)

        # lru_cache saves the output of this function so it doesn't read the file again
        # if it was recently called.
        @lru_cache(maxsize=MAX_CACHE)
        def _read_file_data(self, file_path):
            # PPFromFile is my "all-in-one" solution for preprocessing.
            data = PPFromFile(data_filepath = file_path,
                              state_column = "Channels",
                              current_column = "Noisy Current",
                              max_states = self.max_states,
                              window_size = self.window_size,
                              batch_size = self.batch_size,
                              state_dict = self.state_dict,
                              tts = False)
            return (data.x_data, data.y_data)

        # __len__ method required for generators.
        def __len__(self):
            return ceil(self.no_lines))

        # Use the functions above to find the index and file name of the data within the dataset,
        # and use that to read the data correctly.
        def __getitem__(self, idx):
            in_file_idx, file_path = self._find_file_path(idx)
            x_data, y_data = self._read_file_data(file_path)
            return (x_data[in_file_idx], y_data[in_file_idx])

