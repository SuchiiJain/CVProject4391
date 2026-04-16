# ANOTHER AI BOILERPLATE FOR THE TRAINING DATA!!! NEEDS SPECIFIC WORK!!!
# Gemini Pro my beloved
# python3 Train_Model.py to run

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from ASL_Model import ASLSequenceInterpreter # Imports your model architecture

# --- CONFIGURATION ---
DATA_PATH = os.path.join('ASL_Dataset')
# List the exact letters your team recorded. Order matters!
actions = np.array(['A', 'B', 'C']) 
sequence_length = 16

# --- 1. PREPPING THE DATA ---
class ASLDataset(Dataset):
    def __init__(self, data_path, actions):
        self.sequences = []
        self.labels = []
        
        # Loop through every letter folder (A, B, C...)
        for label_num, action in enumerate(actions):
            action_path = os.path.join(data_path, action)
            if not os.path.exists(action_path):
                continue
                
            # Loop through all the .npy video files in that folder
            for file_name in os.listdir(action_path):
                if file_name.endswith('.npy'):
                    # Load the 16x63 array
                    res = np.load(os.path.join(action_path, file_name))
                    self.sequences.append(res)
                    self.labels.append(label_num) # Label 'A' becomes 0, 'B' becomes 1, etc.
                    
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        # Convert the numpy arrays into PyTorch Tensors
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label

print("Loading dataset...")
dataset = ASLDataset(DATA_PATH, actions)

# Split the data: 80% for training the brain, 20% for testing if it actually learned
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders automatically handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- 2. SETTING UP THE AI ---
# Force PyTorch to use the Jetson's GPU if available!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# Initialize the model and send it to the GPU
model = ASLSequenceInterpreter(num_classes=len(actions)).to(device)

# Loss Function: "CrossEntropyLoss" is standard for multiple-choice classification
criterion = nn.CrossEntropyLoss()
# Optimizer: "Adam" is the algorithm that actually tweaks the math to improve the model
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. THE TRAINING LOOP ---
epochs = 50 # How many times the AI will review the entire dataset

print("Starting training...")
for epoch in range(epochs):
    model.train() # Put model in training mode
    total_loss = 0
    
    # Loop through batches of video sequences
    for sequences, labels in train_loader:
        # Send data to the Jetson's GPU
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Step 1: Clear old calculations
        optimizer.zero_grad()
        
        # Step 2: Make a guess
        predictions = model(sequences)
        
        # Step 3: Calculate how wrong the guess was
        loss = criterion(predictions, labels)
        
        # Step 4: Calculate the math tweaks (backpropagation)
        loss.backward()
        
        # Step 5: Apply the tweaks
        optimizer.step()
        
        total_loss += loss.item()
        
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

# --- 4. SAVING THE FINAL BRAIN ---
torch.save(model.state_dict(), 'asl_model_weights.pth')
print("Training complete! Model saved as 'asl_model_weights.pth'")
