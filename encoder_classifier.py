'''
The model we are building is a Custom AutoEncoder Classifier, which will be used to classify the waves into two classes (0 and 1).
The strategy used wil follow the one used in a Germanium physics paper, i.e.
"Deep learning based pulse shape discrimination for germanium detectors"
that can be find here: https://arxiv.org/abs/1903.01462
    
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------
# DATA LOADING AND PREPROCESSING
# ------------------------------
def dataload(path):
    print(f"This function loads the data from the specified path and returns (in this order): \n"
          " - the dataset"
          " - the loader"
          " - the waves and labels")
    print(f"Loading data from {path}...")
    
    # For the italian excel, csv files are separated with ";", while for the rest of the world you should use ","
    data = pd.read_csv(path, sep=';', skiprows=1, header=None) 
    
    waves_nparray = data.iloc[:, 1:].to_numpy()
    labels_nparray = data.iloc[:, 0].to_numpy()
    # This operation converts the labels from -1 and 1 to 0 and 1, which is the format we need for our classification task.
    labels_nparray = (labels_nparray + 1)/2
    
    waves = torch.tensor(waves_nparray, dtype=torch.float32).unsqueeze(1) 
    # The unsqueeze(1) operation adds a channel dimension to the waves tensor, 
    # making it compatible with the Conv1d layer in the AutoEncoder.
    # This will change the shape of the waves tensor from (N, 500) to (N, 1, 500), where N is the number of waves in the dataset.
    labels = torch.tensor(labels_nparray, dtype=torch.float32)
    
    dataset = TensorDataset(waves, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Data loaded successfully. Waves shape: {waves.size()}, Labels shape: {labels.size()}")
    
    return dataset, loader, waves, labels

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
    

class AutoEncoder(nn.Module):
    '''
    The AutoEncoderstructure will be the following:
        We build firstly a convolutional layer with:
            - 1 input channel (our waveform of 500 points is a 1D signal    )
            - 2 output channels (we want to extract 2 features from the waveform)
            - kernel size of 9 (we want to consider 9 consecutive points of the waveform at a time)
            - padding of 4 (to keep the output size of the two features the same as the input size of 500) 
                padding = (kernel_size - 1) / 2 = (9 - 1) / 2 = 4
        Then we apply a ReLU activation function to introduce non-linearity and allow the model to learn complex patterns
        After that, we will apply a max pooling layer to reduce the size of the output and extract the most important features
        Then we will flatten the output with a fully connected layer into a low-dimensional representation of the waveform, 
        which will be used as input for the classifier.
    '''
        
    def __init__(self):
        # This is needed to initialize the nn.Module class, which is the base class for all neural network modules in PyTorch.
        super().__init__()
        
        self.encoder = nn.Sequential(
            # [INITIAL INPUT] 
            # The tensor shape is (batch_size, channels, length) i.e. (1, 1, 500) for a single waveform of length 500.
            # -> 1 wf, 1 channel (the raw signal), 500 points length.
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=9, padding=4), 
            # [AFTER THE CONVOLUTION] 
            # Shape: (1, 2, 500)
            # -> 2 feature maps (channels) arising from 2 kernels. 
            # The length remains 500 due to padding=4.
            nn.ReLU(),   
            # [AFTER RELU]
            # Shape: (1, 2, 500)
            # -> The dimensions remain the same, negative numbers are converted to zeros.
            nn.MaxPool1d(kernel_size=2, stride=2),
            # [AFTER THE MAX POOLING]
            # Shape: (1, 2, 250)
            # -> The channels remain 2, but the length of the waveform is halved (from 500 to 250) because we group points 2 by 2.
            nn.Flatten(),
            # [AFTER THE FLATTEN]
            # Shape: (1, 500)
            # -> The 3D world is "flattened" into a 1D vector. 
            # The 2 channels of 250 points are merged into a single row of 500 numbers (2 * 250 = 500).
            nn.Linear(in_features=2*250, out_features=64),
            # [AFTER THE LINEAR]
            # Shape: (1, 64)
            # -> The 500 pointsare mathematically compressed into a 64-dimensional latent space.
            nn.ReLU()    
        )
        
        self.decoder = nn.Sequential(
            # The decoder is the mirrored reverse of the encoder. 
            # It takes the 64-dimensional latent space and reconstructs the original waveform.
            
            nn.Linear(in_features=64, out_features=2*250),
            # [AFTER THE LINEAR]
            # Shape: (1, 500)
            # -> The 64-dimensional latent space is mathematically expanded back into a 500-dimensional space.
            nn.Unflatten(dim=1, unflattened_size=(2, 250)),
            # [AFTER THE UNFLATTEN]
            # Shape: (1, 2, 250)
            # -> The 1D vector of 500 numbers is reshaped back into a 3D tensor with 2 channels of 250 points each.
            nn.Upsample(scale_factor=2, mode='nearest'),
            # [AFTER THE UPSAMPLE]
            # Shape: (1, 2, 500)
            # -> The length of the waveform is doubled (from 250 to 500) by repeating each point twice.
            nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=9, padding=4),
            # [AFTER THE CONVOLUTION TRANSPOSE]
            # Shape: (1, 1, 500)
            # At this point we should have reconstructed the original waveform, but it may not be perfect due to the compression and decompression process.
        )
    
    def forward(self, x):
        # Here we define the forward pass ion order to process the "x" data through the encoder and obtain the latent representation.
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed
    
    
    '''
    Useful returnings of the AutoEncoder class:
    - AutoEncoder.encoder() returns the whole sequential block of the encoder
    - AutoEncoder.decoder() returns the whole sequential block of the decoder
    - AutoEncoder.parameters() returns the trained parameters of the whole encoder+decoder
    - AutoEncoder.state_dict() returns a Python dictionary with all the trained parameters 
    - AutoEncoder.train() put the network in training mode
    - AutoEncoder.eval() put the network in test mode
    '''
    




class Classifier(nn.Module):
    
    def __init__(self):
        # This is needed to initialize the nn.Module class, which is the base class for all neural network modules in PyTorch.
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(), 
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1)            
        )
        
        
    def forward(self, x):
        classification = self.classifier(x)
        return classification
    

def autoencoder_loss(reconstructed, original):
    # This function calculates the Mean Squared Error (MSE) loss between the reconstructed waveform and the original waveform.
    # The MSE loss is a common choice for regression tasks and autoencoders, 
    # as it measures the average squared difference between the estimated values and the actual value.
    return nn.MSELoss()(reconstructed, original)

def classifier_loss(predicted, label):
    # This function calculates the Binary Cross Entropy (BCE) loss between the predicted labels and the true labels.
    # The BCE loss is a common choice for binary classification tasks, 
    # as it measures the difference between two probability distributions: 
    # the predicted probabilities and the actual labels.
    return nn.BCEWithLogitsLoss()(predicted, label)


def train_autoencoder(model, loader, epochs):
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        
        # Epoch loss initalize at zero
        epoch_loss = 0.0
        
        for waves, _ in loader:
            
            # Gradients initialised to zero
            optimizer.zero_grad() 
            
            # Forward pass
            latent, reconstructed = model(waves) # Automatically recalls the forward function
            
            # Loss calculation
            loss = autoencoder_loss(reconstructed, waves)
            
            # Backpropagation
            loss.backward()
            
            # Weights upgrade
            optimizer.step()
            
            # Epoch loss calculation
            epoch_loss += loss.item() # 
            
        # Total epoche loss calculation
        average_epoch_loss = epoch_loss/len(loader)
        print(f"Epoca [{epoch+1}/{epochs}] - Loss Media: {average_epoch_loss:.4f}")
            
            
    return model
    




