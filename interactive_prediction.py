import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import ipywidgets as widgets
from google.colab import files

# Import necessary functions from the main code
# These would typically be imported from your modules, but for simplicity we'll define them here
# You should replace these with imports from your actual modules

class ProteinDataset:
    def __init__(self, sequences, labels, input_type='fasta'):
        self.sequences = sequences
        self.labels = labels
        self.input_type = input_type
        
        # Amino acid to index mapping for FASTA sequences
        self.aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20  # X for unknown
        }
        
        # Secondary structure to index mapping
        self.ss_to_idx = {'H': 0, 'E': 1, 'C': 2}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.input_type == 'fasta':
            # One-hot encode the sequence
            encoded = np.zeros((len(sequence), 21))  # 20 amino acids + unknown
            for i, aa in enumerate(sequence):
                if aa in self.aa_to_idx:
                    encoded[i, self.aa_to_idx[aa]] = 1
                else:
                    encoded[i, self.aa_to_idx['X']] = 1  # Unknown amino acid
        elif self.input_type == 'pssm':
            # Min-max normalize PSSM
            encoded = sequence.copy()
            # Normalize each row to [0, 1]
            row_min = encoded.min(axis=1, keepdims=True)
            row_max = encoded.max(axis=1, keepdims=True)
            row_range = row_max - row_min
            row_range[row_range == 0] = 1  # Avoid division by zero
            encoded = (encoded - row_min) / row_range
        else:  # combined
            # Extract FASTA and PSSM
            fasta_seq, pssm_seq = sequence
            
            # One-hot encode FASTA
            fasta_encoded = np.zeros((len(fasta_seq), 21))
            for i, aa in enumerate(fasta_seq):
                if aa in self.aa_to_idx:
                    fasta_encoded[i, self.aa_to_idx[aa]] = 1
                else:
                    fasta_encoded[i, self.aa_to_idx['X']] = 1
            
            # Min-max normalize PSSM
            pssm_encoded = pssm_seq.copy()
            row_min = pssm_encoded.min(axis=1, keepdims=True)
            row_max = pssm_encoded.max(axis=1, keepdims=True)
            row_range = row_max - row_min
            row_range[row_range == 0] = 1
            pssm_encoded = (pssm_encoded - row_min) / row_range
            
            # Concatenate FASTA and PSSM
            encoded = np.concatenate([fasta_encoded, pssm_encoded], axis=1)
            
        # Convert label to index
        if label:  # If label is not empty
            label_idx = [self.ss_to_idx[ss] for ss in label]
        else:
            label_idx = []
        
        return torch.FloatTensor(encoded), torch.LongTensor(label_idx)

def collate_fn(batch):
    """
    Custom collate function for variable-length sequences
    """
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[0].size(0), reverse=True)
    
    # Get sequence lengths
    lengths = [x[0].size(0) for x in batch]
    max_length = lengths[0]
    
    # Get batch size and feature dimension
    batch_size = len(batch)
    feature_dim = batch[0][0].size(1)
    
    # Create padded tensors
    padded_seqs = torch.zeros(batch_size, max_length, feature_dim)
    padded_labels = torch.zeros(batch_size, max_length, dtype=torch.long)
    
    # Fill padded tensors
    for i, (seq, label) in enumerate(batch):
        seq_len = seq.size(0)
        padded_seqs[i, :seq_len, :] = seq
        if label.size(0) > 0:  # If label is not empty
            padded_labels[i, :seq_len] = label
    
    # Create length tensor
    lengths = torch.LongTensor(lengths)
    
    return padded_seqs, padded_labels, lengths

def load_pssm_file(file_path):
    """Load PSSM from CSV file"""
    try:
        return pd.read_csv(file_path).values
    except Exception as e:
        print(f"Error loading PSSM file {file_path}: {e}")
        return np.array([])

def prepare_single_sequence_dataset(sequence, pssm_file_path=None, input_type='fasta'):
    """
    Prepare a dataset for a single sequence
    Args:
        sequence: Protein sequence string
        pssm_file_path: Path to PSSM file (optional)
        input_type: 'fasta', 'pssm', or 'combined'
    Returns:
        Dataset for the sequence
    """
    # Empty label since we're predicting
    labels = ['']  
    
    if input_type == 'fasta':
        sequences = [sequence]
    elif input_type == 'pssm':
        if not pssm_file_path:
            raise ValueError("PSSM file path is required for PSSM input type")
        pssm_sequence = load_pssm_file(pssm_file_path)
        sequences = [pssm_sequence]
    else:  # combined
        if not pssm_file_path:
            raise ValueError("PSSM file path is required for combined input type")
        pssm_sequence = load_pssm_file(pssm_file_path)
        # Ensure PSSM and sequence have the same length
        min_len = min(len(sequence), pssm_sequence.shape[0])
        sequence = sequence[:min_len]
        pssm_sequence = pssm_sequence[:min_len, :]
        sequences = [(sequence, pssm_sequence)]
    
    # Create dataset
    dataset = ProteinDataset(sequences, labels, input_type=input_type)
    
    return dataset

def predict_secondary_structure(model, dataset, device):
    """
    Predict the secondary structure using the loaded model
    Args:
        model: Loaded model
        dataset: Dataset containing the sequence
        device: Device to run the model on
    Returns:
        Predicted secondary structure string
    """
    from torch.utils.data import DataLoader
    
    model.to(device)
    model.eval()
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Predict
    with torch.no_grad():
        for data, _, lengths in loader:
            data = data.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            output = model(data, lengths)
            
            # Get predictions
            _, predicted = torch.max(output.squeeze(0), 1)
            
            # Convert to secondary structure string
            idx_to_ss = {0: 'H', 1: 'E', 2: 'C'}
            ss_prediction = ''.join([idx_to_ss[idx.item()] for idx in predicted])
            
            return ss_prediction
    
    return ""

def visualize_prediction(sequence, ss_prediction):
    """
    Visualize the predicted secondary structure
    Args:
        sequence: Amino acid sequence
        ss_prediction: Predicted secondary structure
    """
    plt.figure(figsize=(15, 3))
    
    # Plot sequence
    for i, aa in enumerate(sequence):
        plt.text(i, 0.5, aa, ha='center', va='center', fontsize=8)
    
    # Plot secondary structure with colors
    for i, ss in enumerate(ss_prediction):
        if ss == 'H':
            color = 'red'
        elif ss == 'E':
            color = 'blue'
        else:  # 'C'
            color = 'green'
        plt.text(i, 0, ss, ha='center', va='center', fontsize=8, color=color, weight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Helix (H)'),
        Patch(facecolor='blue', label='Strand (E)'),
        Patch(facecolor='green', label='Coil (C)')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.xlim(-1, len(sequence))
    plt.ylim(-0.5, 1)
    plt.axis('off')
    plt.title('Protein Secondary Structure Prediction')
    plt.tight_layout()
    plt.show()

def load_model(model_path, device):
    """
    Load a model from a checkpoint file
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
    Returns:
        Loaded model
    """
    from model import PSSPModel
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparameters']
    
    # Create model
    model = PSSPModel(
        input_dim=hyperparams['input_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        output_dim=3,
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def load_models_from_drive(drive_dir='/content/drive/MyDrive/protein_ss_prediction/models', target_dir='models'):
    """
    Load models from Google Drive
    Args:
        drive_dir: Source directory in Google Drive
        target_dir: Target directory to save models locally
    Returns:
        List of available model paths
    """
    # Import Google Drive mounting tools
    try:
        from google.colab import drive
        print("Google Colab detected. Mounting Google Drive...")
        drive.mount('/content/drive')
    except ImportError:
        print("Not running in Google Colab or drive module not available.")
        return []
    
    # Check if drive directory exists
    if not os.path.exists(drive_dir):
        print(f"Drive directory {drive_dir} does not exist.")
        return []
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy model files from Google Drive
    model_files = [f for f in os.listdir(drive_dir) if f.endswith('.pt')]
    if not model_files:
        print(f"No model files found in {drive_dir}")
        return []
    
    models_available = []
    for model_file in model_files:
        source_path = os.path.join(drive_dir, model_file)
        target_path = os.path.join(target_dir, model_file)
        
        # Copy file
        import shutil
        shutil.copy2(source_path, target_path)
        print(f"Copied {source_path} to {target_path}")
        
        # Determine model type from filename
        if 'fasta' in model_file.lower():
            model_type = 'FASTA'
        elif 'pssm' in model_file.lower():
            model_type = 'PSSM'
        elif 'combined' in model_file.lower():
            model_type = 'Combined'
        else:
            model_type = 'Unknown'
        
        models_available.append((model_type, target_path))
    
    print(f"Successfully loaded {len(model_files)} models from Google Drive.")
    return models_available

# Function to upload PSSM file
def upload_pssm():
    """
    Upload a PSSM file
    Returns:
        Path to the uploaded file
    """
    uploaded = files.upload()
    for filename, content in uploaded.items():
        print(f"Uploaded: {filename}")
        return filename
    return None

# Create interactive interface
def create_interactive_interface():
    """
    Create an interactive interface for protein secondary structure prediction
    """
    # Load models
    print("Loading models...")
    models_available = []
    
    # Try to load models from Google Drive
    try:
        models_available = load_models_from_drive()
    except Exception as e:
        print(f"Error loading models from Google Drive: {e}")
    
    # If no models found in Google Drive, check local models
    if not models_available:
        print("Checking local models...")
        if os.path.exists('models'):
            for model_file in os.listdir('models'):
                if model_file.endswith('.pt'):
                    if 'fasta' in model_file.lower():
                        model_type = 'FASTA'
                    elif 'pssm' in model_file.lower():
                        model_type = 'PSSM'
                    elif 'combined' in model_file.lower():
                        model_type = 'Combined'
                    else:
                        model_type = 'Unknown'
                    models_available.append((model_type, os.path.join('models', model_file)))
    
    if not models_available:
        print("No models found. Please train models first.")
        return
    
    print(f"Available models: {[model[0] for model in models_available]}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create input widgets
    sequence_input = widgets.Textarea(
        placeholder='Enter protein sequence here...',
        description='Sequence:',
        layout=widgets.Layout(width='500px', height='100px')
    )
    
    # Create model selection dropdown
    model_options = [model[0] for model in models_available]
    model_dropdown = widgets.Dropdown(
        options=model_options,
        description='Model:',
        disabled=False
    )
    
    # Create PSSM upload button
    pssm_button = widgets.Button(
        description="Upload PSSM",
        disabled=False,
        button_style='',
        tooltip='Upload PSSM file',
        icon='upload'
    )
    
    # Create output widget for PSSM file name
    pssm_output = widgets.Output()
    
    # Create submit button
    submit_button = widgets.Button(
        description="Predict",
        disabled=False,
        button_style='success',
        tooltip='Predict secondary structure',
        icon='check'
    )
    
    # Create output widget for results
    result_output = widgets.Output()
    
    # Variable to store uploaded PSSM file path
    pssm_file_path = [None]  # Use list to allow modification in nested function
    
    # Function to handle PSSM upload
    def handle_pssm_upload(b):
        with pssm_output:
            pssm_output.clear_output()
            print("Uploading PSSM file...")
            pssm_file_path[0] = upload_pssm()
            if pssm_file_path[0]:
                print(f"PSSM file uploaded: {pssm_file_path[0]}")
            else:
                print("No PSSM file uploaded.")
    
    # Function to handle prediction
    def handle_prediction(b):
        with result_output:
            result_output.clear_output()
            
            # Get input values
            sequence = sequence_input.value.strip()
            model_type = model_dropdown.value
            
            if not sequence:
                print("Please enter a protein sequence.")
                return
            
            # Find model path
            model_path = None
            for m_type, m_path in models_available:
                if m_type == model_type:
                    model_path = m_path
                    break
            
            if not model_path:
                print(f"Model {model_type} not found.")
                return
            
            # Check if PSSM is required but not provided
            if (model_type == 'PSSM' or model_type == 'Combined') and not pssm_file_path[0]:
                print(f"PSSM file is required for {model_type} model. Please upload a PSSM file.")
                return
            
            try:
                print(f"Predicting with {model_type} model...")
                
                # Load model
                model = load_model(model_path, device)
                
                # Prepare dataset
                input_type = model_type.lower()
                dataset = prepare_single_sequence_dataset(
                    sequence, 
                    pssm_file_path=pssm_file_path[0], 
                    input_type=input_type
                )
                
                # Predict
                ss_prediction = predict_secondary_structure(model, dataset, device)
                
                # Print results
                print(f"Sequence length: {len(sequence)}")
                print(f"Prediction length: {len(ss_prediction)}")
                
                # Calculate secondary structure composition
                h_count = ss_prediction.count('H')
                e_count = ss_prediction.count('E')
                c_count = ss_prediction.count('C')
                total = len(ss_prediction)
                
                print(f"Helix (H): {h_count} ({h_count/total*100:.1f}%)")
                print(f"Strand (E): {e_count} ({e_count/total*100:.1f}%)")
                print(f"Coil (C): {c_count} ({c_count/total*100:.1f}%)")
                
                # Visualize prediction
                visualize_prediction(sequence[:len(ss_prediction)], ss_prediction)
                
            except Exception as e:
                print(f"Error making prediction: {e}")
                import traceback
                traceback.print_exc()
    
    # Attach event handlers
    pssm_button.on_click(handle_pssm_upload)
    submit_button.on_click(handle_prediction)
    
    # Display the interface
    display(HTML("<h2>Protein Secondary Structure Prediction</h2>"))
    display(sequence_input)
    display(model_dropdown)
    display(pssm_button)
    display(pssm_output)
    display(submit_button)
    display(result_output)

# Run the interactive interface
if __name__ == "__main__":
    create_interactive_interface() 