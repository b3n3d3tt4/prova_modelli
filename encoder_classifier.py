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
import copy
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# --------------------------------------------
# DATA LOADING AND PREPROCESSING for CSV files
# --------------------------------------------
def dataload_csv(path):
    print(f"This function loads the data from the specified path and returns (in this order): \n"
          " - the dataset"
          " - the loader"
          " - the waves and labels")
    print(f"Loading data from {path}...")
    
    # For the italian excel, csv files are separated with ";", while for the rest of the world you should use ","
    data = pd.read_csv(path, sep=';', skiprows=1, header=None) 
    
    waves_nparray = data.iloc[:, 1:501].to_numpy()
    labels_nparray = data.iloc[:, 0].to_numpy()
    # This operation converts the labels from -1 and 1 to 0 and 1, which is the format we need for our classification task.
    labels_nparray = (labels_nparray + 1)/2
    
    waves = torch.tensor(waves_nparray, dtype=torch.float32)
    labels = torch.tensor(labels_nparray, dtype=torch.float32)
    
    mean = waves.mean(dim=1, keepdim=True)
    std = waves.std(dim=1, keepdim=True)
    
    waves = (waves - mean) / (std+1e-9) # Avoiding divisions by zero
    
    # The unsqueeze(1) operation adds a channel dimension to the waves tensor, 
    # making it compatible with the Conv1d layer in the AutoEncoder.
    # This will change the shape of the waves tensor from (N, 500) to (N, 1, 500), where N is the number of waves in the dataset.
    waves = waves.unsqueeze(1)
    
    dataset = TensorDataset(waves, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True) 
    # batch_size is the dimension of the wave packets to analyse
    # shuffle=True allows waveforms mixing
    
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
            nn.Linear(in_features=4000, out_features=128), # 4000 = 32 * 125
            # [AFTER THE LINEAR]
            # Shape: (1, 128)
            # -> The 4000 points are mathematically compressed into a 128-dimensional latent space.
            nn.LeakyReLU(0.1)    
        )
        
        self.decoder = nn.Sequential(
            # The decoder is the mirrored reverse of the encoder. 
            # It takes the 128-dimensional latent space and reconstructs the original waveform.
            
            nn.Linear(in_features=128, out_features=4000),
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
            nn.Linear(in_features=128, out_features=64),
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
    
    # --------------------
    # AUTOENCODER TRAINING
    # --------------------
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # This learning rate is the standard for the Adam alghoritm
    
    for epoch in range(epochs):
        
        # Epoch loss initalize at zero
        epoch_loss = 0.0
        
        for waves, _ in loader:
            
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
    waves, _ = next(iter(loader))
    
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
        for waves, _ in (loader):
            
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
        
    waves, _ = next(iter(loader))
    
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


def train_classifier(autoencoder, classifier, loader, epochs, patience=50): 
    # patience is the maximum number of epochs for which we tolerate that the accuracy doesn't improve
    
    # -------------------------
    # DATASET SPLIT (80% / 20%)
    # -------------------------
    dataset = loader.dataset
    batch_size = loader.batch_size
    
    # Calcoliamo le dimensioni (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    
    # Dividiamo il dataset casualmente
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    
    # Creiamo i due nuovi loader interni
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset splittato: {train_size} sample per il Train, {validation_size} sample per la Validation.\n")
    
    # The autoencoder is in evaluation mode
    autoencoder.eval()
    
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    # weight_decay avoids overfitting because implements regularisation(L2)
    
    # We implement also validation in order to avoid overfitting
    best_validation_loss = float('inf') 
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(classifier.state_dict())
    
    for epoch in range(epochs):
        
        # The classifier is in training mode
        classifier.train()
        
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for waves, labels in train_loader:
            optimizer.zero_grad()
            
            with torch.no_grad():
                latent = autoencoder.encoder(waves)
                
            predictions = classifier(latent)
            labels = labels.unsqueeze(1)
            
            loss = classifier_loss(predictions, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Since we use BCEWithLogitsLoss, raw predictions have to be passed in a sigmoid and round up to get 0 or 1
            predicted_classes = torch.sigmoid(predictions).round()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
            
        train_accuracy = (correct_predictions / total_samples) * 100
        average_train_loss = epoch_loss / len(train_loader)
        
        # -----------------
        # VALIDATION PHASE
        # -----------------
        classifier.eval()
        validation_loss = 0.0
        
        with torch.no_grad():
            for waves, labels in validation_loader:
                latent = autoencoder.encoder(waves)
                predictions = classifier(latent)
                labels = labels.unsqueeze(1)
                loss = classifier_loss(predictions, labels)
                validation_loss += loss.item()
                
        average_validation_loss = validation_loss / len(validation_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] \nTrain Loss: {average_train_loss:.4f} \nValidation Loss: {average_validation_loss:.4f} \nTrain Accuracy: {train_accuracy:.2f}%")
        
        # -----------------
        # EARLY STOPPING
        # -----------------
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            epochs_no_improve = 0
            # Save the best weights
            best_model_weights = copy.deepcopy(classifier.state_dict())
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}! Restoring best weights.")
            break

    # Restore the model to the state where it had the lowest validation loss
    classifier.load_state_dict(best_model_weights)
    return classifier

def test_classifier (autoencoder, classifier, loader):
    
    autoencoder.eval()
    
    classifier.eval()          
        
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for waves, labels in (loader):
            
            # Production of the latent space of the wave
            latent, _ = autoencoder(waves) # Automatically recalls the forward function
            
            # Classification based on the latent space
            classified = classifier(latent)
            
            # Unsqueesing of labels
            labels = labels.unsqueeze(1)
            
            # Loss evaluation for the single wave
            loss = classifier_loss(classified, labels)
            
            # Test loss updating
            test_loss += loss.item()
            
            # Accuracy calculation
            predicted_classes = torch.sigmoid(classified).round()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
            
        average_test_loss = test_loss / len(loader)
        accuracy = (correct_predictions / total_samples) * 100
        
        print(f"Classification - Test result: \nAverage loss: {average_test_loss:.4f} \nAccuracy: {accuracy:.2f}%")
        
    return average_test_loss, accuracy     


def train_masked_autoencoder(model, loader, epochs):
    
    # --------------------
    # AUTOENCODER TRAINING
    # --------------------
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # This learning rate is the standard for the Adam algorithm
    
    for epoch in range(epochs):
        
        # Epoch loss initialized at zero
        epoch_loss = 0.0
        
        for waves, _ in loader:
            
            # Gradients initialized to zero
            optimizer.zero_grad() 
            
            # Time jittering
            shifted_waves = time_shift(waves)
            
            # Block masking application
            shifted_masked_waves, mask = apply_block_mask(shifted_waves, num_blocks=10, block_size=20)
            
            # Noise addition to waves
            noisy_masked_waves = white_noise(shifted_masked_waves)
            
            # Forward pass
            latent, reconstructed = model(noisy_masked_waves) # Automatically recalls the forward function
            
            # Selective Loss calculation: we only compute the error on the hidden blocks!
            loss = autoencoder_loss(reconstructed[mask], shifted_waves[mask])
            
            # Backpropagation
            loss.backward()
            
            # Weights upgrade
            optimizer.step()
            
            # Epoch loss calculation
            epoch_loss += loss.item() 
            
        # Total epoch loss calculation
        average_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average loss: {average_epoch_loss:.4f}")
        
    
    # -----------------------------------------------
    # VISUALISATION OF THE AUTOENCODER RECONSTRUCTION
    # -----------------------------------------------
    
    # We put the model in evaluation mode
    model.eval() 
    
    # We extract 1 of the possible batches from the loader
    waves, _ = next(iter(loader))
    
    # We apply the block mask to the test wave to see how the model fills the gaps!
    masked_waves, mask = apply_block_mask(waves, num_blocks=10, block_size=20)
    
    # We reconstruct the wave through the model with the encoder and decoder
    with torch.no_grad(): # Since at this point we don't want to compute gradients
        _, reconstructed = model(masked_waves)  # Automatically recalls the forward function
    
    # We choose a random index in the batch to select 1 of the 32 samples
    idx = random.randint(0, waves.size(0) - 1)
    
    # We extract data from the single element of the chosen batch and squeeze it into a NumPy array
    original_sample = waves[idx].squeeze().numpy()
    reconstructed_sample = reconstructed[idx].squeeze().numpy()
    
    # We extract the specific mask for this single wave and convert it to NumPy
    sample_mask = mask[idx].squeeze().numpy()
    
    plt.figure(figsize=(10, 5))
    
    # 1. Plot the original wave (slightly transparent to act as a background)
    plt.plot(original_sample, label='Original waveform', color='tab:blue', alpha=0.4)
    
    # 2. Plot the reconstructed wave from the model
    plt.plot(reconstructed_sample, label='Reconstructed waveform', color='tab:orange', linewidth=2)
    
    # 3. HIGHLIGHT THE MISSING ZONES (THE GAPS) IN LIGHT RED
    # We create an array for the X-axis
    x_axis = np.arange(len(original_sample))
    
    # We use fill_between to color the background vertically where the mask is True.
    # alpha=0.2 makes it a transparent light red. 
    # The transform parameter ensures the color spans the entire height of the graph.
    plt.fill_between(x_axis, 0, 1, where=sample_mask, color='red', alpha=0.2, 
                     transform=plt.gca().get_xaxis_transform(), label='Masked Zone')
            
    plt.title('Masked Autoencoder - Block Reconstruction')
    plt.legend()
    plt.show()
            
    return model