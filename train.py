import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pongmodel import PongModel

try:
    data = pd.read_csv("data/pong_data.csv")
except FileNotFoundError:
    print("Data file not found. Please ensure 'pong_data.csv' is in the 'data' directory.")
    exit(1)

# Define features
X = data[["ball_x", "ball_y", "ball_velocity_x", "ball_velocity_y", "player_paddle_y"]]
y = data[["ai_paddle_y"]].values

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)  # Shape (n, 1)
y_val_scaled = y_scaler.transform(y_val)

class PongDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# Create dataloaders
train_dataset = PongDataset(X_train_scaled, y_train_scaled)
val_dataset = PongDataset(X_val_scaled, y_val_scaled)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = PongModel(input_size=X_train.shape[1])

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        sample_inputs, sample_labels = next(iter(val_dataloader))
        sample_outputs = model(sample_inputs)

    # Inverse-transform predictions and labels
    preds_pixels = y_scaler.inverse_transform(sample_outputs.numpy())
    labels_pixels = y_scaler.inverse_transform(sample_labels.numpy())

    # Calculate MAE (Mean Absolute Error) in pixels
    mae = np.mean(np.abs(preds_pixels - labels_pixels))
    print(f"MAE (pixels): {mae:.2f}")

    torch.save(model.state_dict(), "models/pong_ai.pth")
    joblib.dump(scaler, "scalers/scaler.gz")
    joblib.dump(y_scaler, "scalers/y_scaler.gz")