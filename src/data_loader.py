import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import pickle
import time

# CELL 1: Dataset class definition
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, input_type='fasta'):
        """
        Initialize the dataset
        Args:
            sequences: List of sequences (either FASTA or PSSM)
            labels: List of secondary structure labels
            input_type: 'fasta' or 'pssm'
        """
        self.sequences = sequences
        self.labels = labels
        self.input_type = input_type
        
        # Amino acid to index mapping for FASTA sequences
        self.aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
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
            encoded = np.zeros((len(sequence), 20))
            for i, aa in enumerate(sequence):
                if aa in self.aa_to_idx:
                    encoded[i, self.aa_to_idx[aa]] = 1
        else:  # PSSM
            encoded = sequence  # Already in correct format
            
        # Convert label to index
        label_idx = [self.ss_to_idx[ss] for ss in label]
        
        return torch.FloatTensor(encoded), torch.LongTensor(label_idx)

# Custom collate function to handle variable-length sequences
def collate_fn(batch):
    """
    Custom collate function for variable-length sequences
    Args:
        batch: List of (sequence, label) tuples
    Returns:
        Padded sequences, padded labels, sequence lengths
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
        padded_labels[i, :seq_len] = label
    
    # Create length tensor
    lengths = torch.LongTensor(lengths)
    
    return padded_seqs, padded_labels, lengths

# CELL 2: File loading functions
def load_dssp_file(file_path):
    """
    Load secondary structure from DSSP file
    
    In this dataset, DSSP files are in a simplified FASTA-like format:
    - First line is a header starting with '>'
    - Second line contains the secondary structure sequence using H, E, C encoding
    
    H: Helix
    E: Strand
    C: Coil
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header line (starting with '>') and get the secondary structure sequence
        ss_sequence = ""
        for line in lines:
            if line.startswith('>'):
                continue
            ss_sequence += line.strip()
        
        # Validate that we only have H, E, C characters
        for char in ss_sequence:
            if char not in ['H', 'E', 'C']:
                print(f"Warning: Unexpected character '{char}' in DSSP file {file_path}")
        
        if not ss_sequence:
            print(f"Warning: No secondary structure found in {file_path}")
        
        return ss_sequence
    
    except Exception as e:
        print(f"Error loading DSSP file {file_path}: {e}")
        return ""

def load_fasta_file(file_path):
    """Load sequence from FASTA file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header line and join all sequence lines
        sequence = ""
        for line in lines:
            if line.startswith('>'):
                continue
            sequence += line.strip()
        
        return sequence
    
    except Exception as e:
        print(f"Error loading FASTA file {file_path}: {e}")
        return ""

def load_pssm_file(file_path):
    """Load PSSM from CSV file"""
    try:
        return pd.read_csv(file_path).values
    except Exception as e:
        print(f"Error loading PSSM file {file_path}: {e}")
        return np.array([])

def save_processed_data(data_dict, output_path):
    """
    Save processed data to a pickle file
    Args:
        data_dict: Dictionary containing processed data
        output_path: Path to save the pickle file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Processed data saved to {output_path}")

def load_processed_data(input_path):
    """
    Load processed data from a pickle file
    Args:
        input_path: Path to the pickle file
    Returns:
        Dictionary containing processed data
    """
    with open(input_path, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"Processed data loaded from {input_path}")
    return data_dict

# CELL 3: Dataset preparation function
def prepare_dataset(data_dir, input_type='fasta', random_seed=0, batch_size=32, pssm_tolerance=1, 
                   cache_dir=None, use_cache=True, force_reload=False):
    """
    Prepare the dataset by loading all files and splitting into train/val/test
    Args:
        data_dir: Root directory containing dssp/, fasta/, and pssm/ subdirectories
        input_type: 'fasta' or 'pssm'
        random_seed: Random seed for reproducibility
        batch_size: Batch size for data loaders
        pssm_tolerance: Tolerance for PSSM-DSSP length mismatch (only for PSSM input)
        cache_dir: Directory to save/load processed data
        use_cache: Whether to use cached data if available
        force_reload: Whether to force reload data even if cache exists
    Returns:
        train_loader, val_loader, test_loader
    """
    # Set up cache path
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"processed_data_{input_type}_seed{random_seed}_tol{pssm_tolerance}.pkl")
    
    # Check if cached data exists and should be used
    if use_cache and os.path.exists(cache_file) and not force_reload:
        print(f"Loading cached processed data from {cache_file}")
        data_dict = load_processed_data(cache_file)
        
        # Create datasets from cached data
        train_dataset = ProteinDataset(
            data_dict['train_sequences'],
            data_dict['train_labels'],
            input_type
        )
        val_dataset = ProteinDataset(
            data_dict['val_sequences'],
            data_dict['val_labels'],
            input_type
        )
        test_dataset = ProteinDataset(
            data_dict['test_sequences'],
            data_dict['test_labels'],
            input_type
        )
        
        # Create data loaders with custom collate function
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    # If no cache or force reload, process the data
    print(f"Processing data from {data_dir}")
    start_time = time.time()
    
    # Get all file IDs
    dssp_dir = os.path.join(data_dir, 'dssp')
    
    # Fix file ID parsing to preserve the full identifier
    file_ids = []
    for filename in os.listdir(dssp_dir):
        if filename.endswith('.dssp'):
            # Remove only the .dssp extension, keep the rest of the filename
            file_id = filename[:-5]  # Remove '.dssp'
            file_ids.append(file_id)
    
    print(f"Found {len(file_ids)} DSSP files")
    
    # Load all data
    sequences = []
    labels = []
    
    # Track files that were successfully loaded
    valid_files = []
    skipped_files = []
    adjusted_files = []
    
    for file_id in file_ids:
        try:
            # Load label
            dssp_path = os.path.join(dssp_dir, f"{file_id}.dssp")
            label = load_dssp_file(dssp_path)
            
            if not label:
                print(f"Skipping {file_id}: Empty secondary structure")
                skipped_files.append((file_id, "empty_label"))
                continue
            
            # Load sequence
            if input_type == 'fasta':
                fasta_path = os.path.join(data_dir, 'fasta', f"{file_id}.fasta")
                if not os.path.exists(fasta_path):
                    print(f"Skipping {file_id}: FASTA file not found")
                    skipped_files.append((file_id, "fasta_not_found"))
                    continue
                sequence = load_fasta_file(fasta_path)
                if not sequence:
                    print(f"Skipping {file_id}: Empty FASTA sequence")
                    skipped_files.append((file_id, "empty_sequence"))
                    continue
                
                # Ensure sequence and label have the same length
                if len(sequence) != len(label):
                    print(f"Warning: Length mismatch for {file_id}: seq={len(sequence)}, label={len(label)}")
                    print(f"Sequence start: {sequence[:10]}")
                    print(f"Label start: {label[:10]}")
                    skipped_files.append((file_id, "length_mismatch"))
                    continue
            else:  # PSSM
                pssm_path = os.path.join(data_dir, 'pssm', f"{file_id}.csv")
                if not os.path.exists(pssm_path):
                    print(f"Skipping {file_id}: PSSM file not found")
                    skipped_files.append((file_id, "pssm_not_found"))
                    continue
                sequence = load_pssm_file(pssm_path)
                if sequence.size == 0:
                    print(f"Skipping {file_id}: Empty PSSM matrix")
                    skipped_files.append((file_id, "empty_pssm"))
                    continue
                
                # Handle PSSM-DSSP length mismatch with tolerance
                length_diff = abs(sequence.shape[0] - len(label))
                if length_diff > 0:
                    if length_diff <= pssm_tolerance:
                        # Adjust the longer one to match the shorter one
                        if sequence.shape[0] > len(label):
                            # Trim PSSM to match DSSP length
                            sequence = sequence[:len(label)]
                            print(f"Adjusted PSSM length for {file_id}: trimmed {length_diff} residue(s)")
                        else:
                            # Trim DSSP to match PSSM length
                            label = label[:sequence.shape[0]]
                            print(f"Adjusted DSSP length for {file_id}: trimmed {length_diff} residue(s)")
                        adjusted_files.append((file_id, length_diff))
                    else:
                        print(f"Warning: Length mismatch for {file_id}: seq={sequence.shape[0]}, label={len(label)}")
                        skipped_files.append((file_id, "length_mismatch"))
                        continue
                
            sequences.append(sequence)
            labels.append(label)
            valid_files.append(file_id)
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            skipped_files.append((file_id, "error"))
    
    print(f"Successfully loaded {len(valid_files)} files out of {len(file_ids)}")
    
    # Analyze skipped files
    if skipped_files:
        reasons = {}
        for _, reason in skipped_files:
            reasons[reason] = reasons.get(reason, 0) + 1
        print("Skipped files by reason:")
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")
    
    # Report adjusted files
    if adjusted_files:
        print(f"Adjusted {len(adjusted_files)} files for length mismatches within tolerance:")
        for file_id, diff in adjusted_files[:5]:  # Show first 5
            print(f"  {file_id}: adjusted by {diff} residue(s)")
        if len(adjusted_files) > 5:
            print(f"  ... and {len(adjusted_files) - 5} more")
    
    if not valid_files:
        raise ValueError("No valid files were loaded. Check your dataset paths and file formats.")
    
    # Split dataset
    train_idx, temp_idx = train_test_split(
        range(len(sequences)), test_size=0.3, random_state=random_seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=random_seed
    )
    
    print(f"Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Prepare data for caching
    train_sequences = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_sequences = [sequences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    # Save processed data to cache
    if use_cache:
        data_dict = {
            'train_sequences': train_sequences,
            'train_labels': train_labels,
            'val_sequences': val_sequences,
            'val_labels': val_labels,
            'test_sequences': test_sequences,
            'test_labels': test_labels,
            'input_type': input_type,
            'random_seed': random_seed,
            'pssm_tolerance': pssm_tolerance,
            'valid_files': valid_files,
            'skipped_files': skipped_files,
            'adjusted_files': adjusted_files
        }
        save_processed_data(data_dict, cache_file)
    
    # Create datasets
    train_dataset = ProteinDataset(train_sequences, train_labels, input_type)
    val_dataset = ProteinDataset(val_sequences, val_labels, input_type)
    test_dataset = ProteinDataset(test_sequences, test_labels, input_type)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    end_time = time.time()
    print(f"Data processing completed in {end_time - start_time:.2f} seconds")
    
    return train_loader, val_loader, test_loader 