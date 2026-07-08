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
            # -------------------------
            # FIRST CONVOLUTIONAL LAYER
            # -------------------------
            # [INITIAL INPUT] 
            # The tensor shape is (batch_size, channels, length) i.e. (1, 1, 500) for a single waveform of length 500.
            # -> 1 wf, 1 channel (the raw signal), 500 points length.
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2), 
            # [AFTER THE CONVOLUTION] 
            # Shape: (1, 16, 500)
            # -> 16 feature maps (channels) arising from 16 kernels. 
            # The length remains 500 due to padding=2.
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1), # it is better than ReLU for physical signals for a small gradient also negative values are accepted
            # [AFTER LEAKY RELU]
            # Shape: (1, 16, 500)
            # -> The dimensions remain the same, negative numbers are multiplied by a small slope (0.1).
            nn.MaxPool1d(kernel_size=2, stride=2),
            # [AFTER THE MAX POOLING]
            # Shape: (1, 16, 250)
            # -> The channels remain 16, but the length of the waveform is halved (from 500 to 250) because we group points 2 by 2.
            
            
            # ---------------------------
            # SECOND CONVOLUTIONAL LAYER
            # --------------------------
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2), 
            # [AFTER THE CONVOLUTION] 
            # Shape: (1, 32, 250)
            # -> 32 feature maps (channels) arising from 32 kernels. 
            # The length remains 250 due to padding=2.
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1), # it is better than ReLU for physical signals for a small gradient also negative values are accepted
            # [AFTER LEAKY RELU]
            # Shape: (1, 32, 250)
            # -> The dimensions remain the same, negative numbers are multiplied by a small slope (0.1).
            nn.MaxPool1d(kernel_size=2, stride=2),
            # [AFTER THE MAX POOLING]
            # Shape: (1, 32, 125)
            # -> The channels remain 32, but the length of the waveform is halved again (from 250 to 125) because we group points 2 by 2.
            
            
            nn.Flatten(),
            # [AFTER THE FLATTEN]
            # Shape: (1, 4000)
            # -> The 3D tensor is "flattened" into a 1D vector. 
            # The 32 channels of 125 points are merged into a single row of 4000 numbers (32 * 125 = 4000).
            nn.Linear(in_features=4000, out_features=256), # 4000 = 32 * 125
            # [AFTER THE LINEAR]
            # Shape: (1, 256)
            # -> The 4000 points are mathematically compressed into a 128-dimensional latent space.
            nn.LeakyReLU(0.1)    
        )
        
        self.decoder = nn.Sequential(
            # The decoder is the mirrored reverse of the encoder. 
            # It takes the 128-dimensional latent space and reconstructs the original waveform.
            
            nn.Linear(in_features=256, out_features=4000),
            # [AFTER THE LINEAR]
            # Shape: (1, 4000)
            # -> The 128-dimensional latent space is mathematically expanded back into a 4000-dimensional space.
            nn.LeakyReLU(0.1),
            
            nn.Unflatten(dim=1, unflattened_size=(32, 125)),
            # [AFTER THE UNFLATTEN]
            # Shape: (1, 32, 125)
            # -> The 1D vector of 4000 numbers is reshaped back into a 3D tensor with 32 channels of 125 points each.
            
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            # [AFTER THE FIRST CONVOLUTION TRANSPOSE AND LEAKY RELU]
            # Shape: (1, 16, 250)
            # -> The transposed convolution learns how to upsample the length of the waveform from 125 to 250, 
            # while reducing the channels from 32 to 16. Negative values are multiplied by a 0.1 slope.
            
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),            
            # [AFTER THE SECOND CONVOLUTION TRANSPOSE]
            # Shape: (1, 1, 500)
            # -> The transposed convolution learns how to upsample the length of the waveform from 250 to 500,
            # returning to the single initial channel. 
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
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(num_features=64), # Normalisaation to regularise against overfitting
            nn.LeakyReLU(0.1), 
            nn.Dropout(0.3), # Turns off casually the 30% of the training neurons, 
            # while when in test mode (classifier.eval()) all of the neurons are alive
            
            nn.Linear(in_features=64, out_features=16),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(in_features=16, out_features=1)            
        )
        
        
    def forward(self, x):
        classification = self.classifier(x)
        return classification