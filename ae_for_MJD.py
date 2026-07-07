import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# --------------------------------------------
# DATA LOADING AND PREPROCESSING for CSV files
# --------------------------------------------

def dataload_h5py(path):
    print(f"This function loads the data from the specified HDF5 path and returns (in this order): \n"
          " - the dataset\n"
          " - the loader\n"
          " - the waves")
    print(f"Loading data from {path}...")
    
    # Apriamo il file h5 in modalità lettura
    with h5py.File(path, 'r') as f:
        # Estraiamo direttamente il dataset 'raw_waveform'
        # L'operatore [:] carica i dati in memoria come un array NumPy
        waves_nparray = f['raw_waveform'][:]
    
    # Convertiamo in tensore PyTorch
    waves = torch.tensor(waves_nparray, dtype=torch.float32)
    
    # Normalizzazione z-score per ogni forma d'onda
    mean = waves.mean(dim=1, keepdim=True)
    std = waves.std(dim=1, keepdim=True)
    waves = (waves - mean) / (std + 1e-9) # Avoiding divisions by zero
    
    # Aggiungiamo la dimensione del canale per Conv1d -> shape: (N, 1, sequence_length)
    waves = waves.unsqueeze(1)
    
    # Creiamo il dataset usando SOLO le waves
    dataset = TensorDataset(waves)
    loader = DataLoader(dataset, batch_size=32, shuffle=True) 
    
    print(f"Data loaded successfully. Waves shape: {waves.size()}")
    
    return dataset, loader, waves

# --------------------------
# WAVEFUNCTION VISUALISATION
# --------------------------
def plot_wave(wave_array):
    print(f"This function plots the wave from the provided torch tensor.")
    
    n = random.choice(range(len(wave_array)))
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(wave_array[n][0])), wave_array[n][0], color='orange', alpha=0.7, lw=0.8)
    plt.scatter(range(len(wave_array[n][0])), wave_array[n][0], s=10)
    plt.title(f"Wave number {n}")
    plt.show()
    
    
# --------------------
# WHITE NOISE ADDITION
# --------------------

def white_noise(waves, noise_level=0.05):
    
    noise = torch.randn_like(waves) * noise_level # Gaussian-like white noise
    
    noisy_waves = waves + noise
    
    return noisy_waves

# --------------
# TIME JITTERING
# --------------
def time_shift(waves, max_shift=15):
    
    batch_size, channels, lenght = waves.shape
    
    shifted_waves = torch.zeros_like(waves)
    
    for i in range(batch_size):
        shift = random.randint(-max_shift, +max_shift)
        
        if shift > 0:
            shifted_waves[i, :, shift:] = waves[i, :, -shift] # moves the waveform to the right
            shifted_waves[i, :, :shift] = waves[i, :, 0:1] # fill the left empty part with the first value (the baseline)
            
        elif shift < 0: # this is exactly the same but with left and right inverted
            shifted_waves[i, :, :shift] = waves[i, :, -shift:]
            shifted_waves[i, :, shift:] = waves[i, :, -1:]
            
        else:
            shifted_waves[i] = waves[i]
            
    return shifted_waves

# -------------
# MASK FUNCTION
# -------------
def apply_block_mask(waves, num_blocks=10, block_size=20):
    
    batch_size, channels, length = waves.shape
    
    # Create an initially all-False mask (no missing points at the beginning)
    mask = torch.zeros_like(waves, dtype=torch.bool, device=waves.device)
    
    for i in range(batch_size):
        # Generate 'num_blocks' random starting points.
        # We use (length - block_size) as the upper limit (e.g., 500 - 20 = 480) 
        # to prevent the block from exceeding the waveform's length.
        start_indices = torch.randint(0, length - block_size + 1, (num_blocks,))
        
        # For each generated starting point, set the mask to True for 'block_size' consecutive points
        for start in start_indices:
            
            mask[i, 0, start : start + block_size] = True
            
    # Clone the original waves and apply 0.0 where the mask is True
    masked_waves = waves.clone()
    masked_waves[mask] = 0.0
    
    return masked_waves, mask


class AutoEncoderHDF5(nn.Module):
    '''
    The AutoEncoder structure will be the following:
        We build firstly a convolutional layer with:
            - 1 input channel (our waveform of 3800 points is a 1D signal)
            - 16 output channels (we want to extract 16 features from the waveform)
            - kernel size of 5 (we want to consider 5 consecutive points of the waveform at a time)
            - padding of 2 (to keep the output size of the features the same as the input size of 3800) 
                padding = (kernel_size - 1) / 2 = (5 - 1) / 2 = 2
        Then we apply a LeakyReLU activation function to introduce non-linearity and allow the model to learn complex patterns.
        After that, we will apply a max pooling layer to reduce the size of the output and extract the most important features.
        Then we will flatten the output with a fully connected layer into a low-dimensional representation of the waveform, 
        which will be used as input for the latent space analysis.
    '''
        
    def __init__(self):
        # This is needed to initialize the nn.Module class, which is the base class for all neural network modules in PyTorch.
        super().__init__()
        
        self.encoder = nn.Sequential(
            # -------------------------
            # FIRST CONVOLUTIONAL LAYER
            # -------------------------
            # [INITIAL INPUT] 
            # The tensor shape is (batch_size, channels, length).
            # -> 1 channel (the raw signal), 3800 points length.
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2), 
            # [AFTER THE CONVOLUTION] 
            # Shape: (batch_size, 16, 3800)
            # -> 16 feature maps (channels) arising from 16 kernels. 
            # The length remains 3800 due to padding=2.
            nn.LeakyReLU(0.1), # it is better than ReLU for physical signals for a small gradient also negative values are accepted
            # [AFTER LEAKY RELU]
            # Shape: (batch_size, 16, 3800)
            # -> The dimensions remain the same, negative numbers are multiplied by a small slope (0.1).
            nn.MaxPool1d(kernel_size=2, stride=2),
            # [AFTER THE MAX POOLING]
            # Shape: (batch_size, 16, 1900)
            # -> The channels remain 16, but the length of the waveform is halved (from 3800 to 1900) because we group points 2 by 2.
            
            
            # ---------------------------
            # SECOND CONVOLUTIONAL LAYER
            # --------------------------
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2), 
            # [AFTER THE CONVOLUTION] 
            # Shape: (batch_size, 32, 1900)
            # -> 32 feature maps (channels) arising from 32 kernels. 
            # The length remains 1900 due to padding=2.
            nn.LeakyReLU(0.1), 
            # [AFTER LEAKY RELU]
            # Shape: (batch_size, 32, 1900)
            nn.MaxPool1d(kernel_size=2, stride=2),
            # [AFTER THE MAX POOLING]
            # Shape: (batch_size, 32, 950)
            # -> The channels remain 32, but the length of the waveform is halved again (from 1900 to 950) because we group points 2 by 2.
            
            
            nn.Flatten(),
            # [AFTER THE FLATTEN]
            # Shape: (batch_size, 30400)
            # -> The 3D tensor is "flattened" into a 1D vector. 
            # The 32 channels of 950 points are merged into a single row of 30400 numbers (32 * 950 = 30400).
            nn.Linear(in_features=30400, out_features=128),
            # [AFTER THE LINEAR]
            # Shape: (batch_size, 128)
            # -> The 30400 points are mathematically compressed into a 128-dimensional latent space.
            nn.LeakyReLU(0.1)    
        )
        
        self.decoder = nn.Sequential(
            # The decoder is the mirrored reverse of the encoder. 
            # It takes the 128-dimensional latent space and reconstructs the original waveform.
            
            nn.Linear(in_features=128, out_features=30400),
            # [AFTER THE LINEAR]
            # Shape: (batch_size, 30400)
            # -> The 128-dimensional latent space is mathematically expanded back into a 30400-dimensional space.
            nn.LeakyReLU(0.1),
            
            nn.Unflatten(dim=1, unflattened_size=(32, 950)),
            # [AFTER THE UNFLATTEN]
            # Shape: (batch_size, 32, 950)
            # -> The 1D vector of 30400 numbers is reshaped back into a 3D tensor with 32 channels of 950 points each.
            
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            # [AFTER THE FIRST CONVOLUTION TRANSPOSE AND LEAKY RELU]
            # Shape: (batch_size, 16, 1900)
            # -> The transposed convolution learns how to upsample the length of the waveform from 950 to 1900, 
            # while reducing the channels from 32 to 16. Negative values are multiplied by a 0.1 slope.
            
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),            
            # [AFTER THE SECOND CONVOLUTION TRANSPOSE]
            # Shape: (batch_size, 1, 3800)
            # -> The transposed convolution learns how to upsample the length of the waveform from 1900 to 3800,
            # returning to the single initial channel. 
            # At this point we should have reconstructed the original waveform.
        )
    
    def forward(self, x):
        # Here we define the forward pass in order to process the "x" data through the encoder and obtain the latent representation.
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed



def autoencoder_loss(reconstructed, original):
    # This function calculates the Mean Squared Error (MSE) loss between the reconstructed waveform and the original waveform.
    # The MSE loss is a common choice for regression tasks and autoencoders, 
    # as it measures the average squared difference between the estimated values and the actual value.
    return nn.MSELoss()(reconstructed, original)


def train_autoencoder(model, loader, epochs):
    
    # --------------------
    # AUTOENCODER TRAINING
    # --------------------
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # This learning rate is the standard for the Adam alghoritm
    
    for epoch in range(epochs):
        
        # Epoch loss initalize at zero
        epoch_loss = 0.0
        
        for (waves, ) in loader: # the (waves, ) syntax is essential for the correct extraction of data from the loader
            
            # Gradients initialised to zero
            optimizer.zero_grad() 
            
            # Time jittering
            shifted_waves = time_shift(waves)
            
            # Noise addition to waves
            noisy_waves = white_noise(shifted_waves)
            
            # Forward pass
            latent, reconstructed = model(noisy_waves) # Automatically recalls the forward function
            
            # Loss calculation
            loss = autoencoder_loss(reconstructed, shifted_waves)
            
            # Backpropagation
            loss.backward()
            
            # Weights upgrade
            optimizer.step()
            
            # Epoch loss calculation
            epoch_loss += loss.item() 
            
        # Total epoch loss calculation
        average_epoch_loss = epoch_loss/len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average loss: {average_epoch_loss:.4f}")
        
    
    
    # -----------------------------------------------
    # VISUALISATION OF THE AUTOENCODER RECONSTRUCTION
    # -----------------------------------------------
    # We put the model in evaluation mode
    model.eval() 
    
    # We extract 1 of the possible batches from the loader
    (waves, ) = next(iter(loader))
    
    # We reconstruct the wave through the model with the encoder and decoder
    with torch.no_grad(): # Since at this point we don't want to compute gradients
        _, reconstructed = model(waves)  # Automatically recalls the forward function
    
    # We choose a random index in the batch to selcetc 1 of the 32 samples in the chosen batch
    idx = random.randint(0, waves.size(0) - 1)
    
    # We extract data from the single element of the chosen bax and squeeze it into a NumPy array
    original_sample = waves[idx].squeeze().numpy()
    reconstructed_sample = reconstructed[idx].squeeze().numpy()
    
    plt.figure(figsize=(9, 4))
    plt.plot(original_sample, label='Original waveform')
    plt.plot(reconstructed_sample, label='Reconstructed waveform')            
    plt.title('Original vs Reconstructed comparison - Training set')
    plt.legend()
    plt.show()
            
    return model


def test_autoencoder(model, loader):
    
    # ----------------
    # AUTOENCODER TEST
    # ----------------
    
    model.eval()
    
    test_loss = 0.0
    
    with torch.no_grad():
        for (waves, ) in (loader):
            
            # Recontruction of the wave
            _, reconstructed = model(waves) # Automatically recalls the forward function
            
            # Loss evaluation for the single wave
            loss = autoencoder_loss(reconstructed, waves)
            
            # Test loss updating
            test_loss += loss.item()
            
        average_test_loss = test_loss / len(loader)
        
        print(f"Recontruction - Test result: Average loss: {average_test_loss:.4f}")
        
        
    # -----------------------------------------------
    # VISUALISATION OF THE AUTOENCODER RECONSTRUCTION
    # -----------------------------------------------
        
    (waves, ) = next(iter(loader))
    
    with torch.no_grad():
        _, reconstructed = model(waves)
        
    idx = random.randint(0, waves.size(0) - 1)
    
    original_sample = waves[idx].squeeze().numpy()
    reconstructed_sample = reconstructed[idx].squeeze().numpy()
    
    plt.figure(figsize=(9, 4))
    plt.plot(original_sample, label='Original waveform')
    plt.plot(reconstructed_sample, label='Reconstructed waveform')            
    plt.title('Original vs Reconstructed comparison - Test set')
    plt.legend()
    plt.show()
    
              
    return average_test_loss