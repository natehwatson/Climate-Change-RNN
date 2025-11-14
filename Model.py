import pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn, torch.optim as optim
import time
#
# LSTM Recurrent neural network
#

class LSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, num_layers:int=1,dropout=0.3) -> None:
        '''
        Args:
        - input_size: number of features in input vector
        - hidden_size: number of hidden neurons
        - output_size: number of features in your output vector
        - num_layers: number of LSTM layers
        '''
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward propagation through the LSTM network

        Args:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
        - Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Take the last time step
        out = self.fc(out[:, -1, :])

        return out


class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for LSTM time series data
    """
    def __init__(self, data, sequence_length=5, predict_steps=10):
        """
        Args:
        - data: Pandas DataFrame with time series features
        - sequence_length: Number of previous time steps to use for prediction
        - predict_steps: Number of future steps to predict
        """
        self.data = data
        self.sequence_length = sequence_length
        self.predict_steps = predict_steps

        self.features = data[['Coal_Production_change(%)','Aviation_passanger_growth(%)']].values
        self.targets = data['Nasa_Temp_Anomaly'].values

        self.scaler = MinMaxScaler()
        self.normalized_features = self.scaler.fit_transform(self.features)

        self.sequences = []
        self.labels = []
        self.create_sequences()

    def to_string():
        return data.to_string()

    def create_sequences(self):
        """
        Create input sequences and corresponding labels
        """
        for i in range(len(self.normalized_features) - self.sequence_length - self.predict_steps + 1):
            seq = self.normalized_features[i:i+self.sequence_length]

            label = self.targets[i+self.sequence_length+self.predict_steps-1]

            self.sequences.append(seq)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self,idx):
        """
        Convert sequences to PyTorch tensors
        """
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


def generate_time_series_data(data, num_samples=1000):
    """
   data: dataset to be used
   num_samples = number of samples in the dataset
    """


    years = data['Year']

    feature_data = data[['Coal_Production_change(%)','Nasa_Temp_Anomaly', 'Aviation_passanger_growth(%)']]
    columns = feature_data.columns
    feature_data = feature_data.to_numpy()

    df = pd.DataFrame(data=feature_data, index=years, columns=columns)


    # Create target: a synthetic time series with some complexity
    df['Target'] = data['Objective'].to_numpy()

    return df

def prepare_data_loaders(dataset, batch_size=32, sequence_length=20, predict_steps=1):
    """
    Prepare training and validation data loaders
    """
    full_dataset = dataset

    train_size = int(0.8 * len(full_dataset))
    train_data = full_dataset.iloc[:train_size]
    val_data = full_dataset.iloc[train_size:]

    train_dataset = TimeSeriesDataset(
        train_data,
        sequence_length=sequence_length,
        predict_steps=predict_steps
    )
    val_dataset = TimeSeriesDataset(
        val_data,
        sequence_length=sequence_length,
        predict_steps=predict_steps
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device=None):
    """
    Train the LSTM model

    Args:
    - model: LSTM neural network model
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - criterion: Loss function
    - optimizer: Optimization algorithm
    - num_epochs: Number of training epochs
    - device: Computing device (CPU/GPU)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    train_losses = []
    val_losses = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.squeeze())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)
                val_loss = criterion(outputs, batch_labels.squeeze())
                epoch_val_loss += val_loss.item()

        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses

def plot_training_history(train_losses, val_losses):
    """
    Plot training and validation losses
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_with_model(model, input_sequence, scaler=None):
    """
    Make predictions with the trained model

    Args:
    - model: Trained LSTM model
    - input_sequence: Input sequence for prediction
    - scaler: Optional scaler for inverse transformation

    Returns:
    - Predicted value
    """
    model.eval()
    with torch.no_grad():
        # Convert input to tensor if not already
        if not isinstance(input_sequence, torch.Tensor):
            input_sequence = torch.FloatTensor(input_sequence)

        # Add batch dimension if missing
        if input_sequence.dim() == 2:
            input_sequence = input_sequence.unsqueeze(0)

        prediction = model(input_sequence)

        # Inverse transform if scaler is provided
        if scaler is not None:
            prediction = scaler.inverse_transform(prediction.numpy())

        return prediction


