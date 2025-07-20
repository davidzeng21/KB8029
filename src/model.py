import torch
import torch.nn as nn

# CELL 1: Model architecture definition
class SecondaryStructurePredictor(nn.Module):
    def __init__(self, input_channels=20, hidden_size=128, num_layers=2, dropout=0.3):
        """
        Initialize the model
        Args:
            input_channels: Number of input channels (20 for FASTA, 20 for PSSM)
            hidden_size: Hidden size for LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(SecondaryStructurePredictor, self).__init__()
        
        # CNN for local feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        
        # BiLSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 3)  # 3 classes: H, E, C
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_channels)
        
        # CNN layers
        x = x.transpose(1, 2)  # (batch_size, input_channels, seq_len)
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = x.transpose(1, 2)  # (batch_size, seq_len, channels)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Output layer
        out = self.dropout(context)
        out = self.fc(out)
        
        return out
    
    def predict_sequence(self, x):
        """
        Predict secondary structure for a single sequence
        Args:
            x: Input tensor of shape (1, seq_len, input_channels)
        Returns:
            Predicted secondary structure indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()

# CELL 2: Model creation function
def create_model(input_type='fasta', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create and initialize the model
    Args:
        input_type: 'fasta' or 'pssm'
        device: Device to run the model on
    Returns:
        Initialized model
    """
    # Both FASTA and PSSM inputs have 20 channels
    model = SecondaryStructurePredictor(input_channels=20)
    model = model.to(device)
    return model 