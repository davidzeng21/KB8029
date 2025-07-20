#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset diagnostic tool for protein secondary structure prediction.
This script checks the dataset for issues and generates a report.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import time
from collections import defaultdict, Counter

def load_dssp_file(file_path):
    """Load secondary structure from DSSP file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header line and join all sequence lines
        ss_sequence = ""
        for line in lines:
            if line.startswith('>'):
                continue
            ss_sequence += line.strip()
        
        # Validate that we only have H, E, C characters
        valid_chars = set(['H', 'E', 'C'])
        invalid_chars = set([char for char in ss_sequence if char not in valid_chars])
        
        if invalid_chars:
            return ss_sequence, f"Invalid characters: {', '.join(invalid_chars)}"
        
        if not ss_sequence:
            return "", "Empty secondary structure"
        
        return ss_sequence, None
    
    except Exception as e:
        return "", f"Error: {str(e)}"

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
        
        # Validate that we only have valid amino acid characters
        valid_aa = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        invalid_chars = set([char for char in sequence if char not in valid_aa])
        
        if invalid_chars:
            return sequence, f"Invalid characters: {', '.join(invalid_chars)}"
        
        if not sequence:
            return "", "Empty sequence"
        
        return sequence, None
    
    except Exception as e:
        return "", f"Error: {str(e)}"

def load_pssm_file(file_path):
    """Load PSSM from CSV file"""
    try:
        pssm = pd.read_csv(file_path).values
        
        if pssm.size == 0:
            return np.array([]), "Empty PSSM matrix"
        
        # Check if PSSM has 20 columns (for 20 amino acids)
        if pssm.shape[1] != 20:
            return pssm, f"Invalid PSSM shape: {pssm.shape}"
        
        return pssm, None
    
    except Exception as e:
        return np.array([]), f"Error: {str(e)}"

def check_dataset(data_dir, output_file=None):
    """
    Check the dataset for issues
    Args:
        data_dir: Root directory containing dssp/, fasta/, and pssm/ subdirectories
        output_file: Path to save the report (CSV)
    Returns:
        DataFrame with results
    """
    # Check if directories exist
    dssp_dir = os.path.join(data_dir, 'dssp')
    fasta_dir = os.path.join(data_dir, 'fasta')
    pssm_dir = os.path.join(data_dir, 'pssm')
    
    if not os.path.exists(dssp_dir):
        print(f"Error: DSSP directory not found at {dssp_dir}")
        return None
    
    if not os.path.exists(fasta_dir):
        print(f"Error: FASTA directory not found at {fasta_dir}")
        return None
    
    if not os.path.exists(pssm_dir):
        print(f"Error: PSSM directory not found at {pssm_dir}")
        return None
    
    # Get all file IDs from DSSP directory
    file_ids = []
    for filename in os.listdir(dssp_dir):
        if filename.endswith('.dssp'):
            # Remove only the .dssp extension, keep the rest of the filename
            file_id = filename[:-5]  # Remove '.dssp'
            file_ids.append(file_id)
    
    print(f"Found {len(file_ids)} DSSP files")
    
    # Initialize results
    results = []
    
    # Statistics
    stats = {
        'total_files': len(file_ids),
        'valid_files': 0,
        'dssp_issues': 0,
        'fasta_issues': 0,
        'pssm_issues': 0,
        'length_mismatches': 0,
        'sequence_lengths': []
    }
    
    # Check each file
    for file_id in file_ids:
        result = {
            'file_id': file_id,
            'dssp_exists': False,
            'fasta_exists': False,
            'pssm_exists': False,
            'dssp_length': 0,
            'fasta_length': 0,
            'pssm_length': 0,
            'dssp_issue': None,
            'fasta_issue': None,
            'pssm_issue': None,
            'length_match': False
        }
        
        # Check DSSP file
        dssp_path = os.path.join(dssp_dir, f"{file_id}.dssp")
        if os.path.exists(dssp_path):
            result['dssp_exists'] = True
            dssp_content, dssp_issue = load_dssp_file(dssp_path)
            result['dssp_issue'] = dssp_issue
            result['dssp_length'] = len(dssp_content)
            
            if dssp_issue:
                stats['dssp_issues'] += 1
        
        # Check FASTA file
        fasta_path = os.path.join(fasta_dir, f"{file_id}.fasta")
        if os.path.exists(fasta_path):
            result['fasta_exists'] = True
            fasta_content, fasta_issue = load_fasta_file(fasta_path)
            result['fasta_issue'] = fasta_issue
            result['fasta_length'] = len(fasta_content)
            
            if fasta_issue:
                stats['fasta_issues'] += 1
        
        # Check PSSM file
        pssm_path = os.path.join(pssm_dir, f"{file_id}.csv")
        if os.path.exists(pssm_path):
            result['pssm_exists'] = True
            pssm_content, pssm_issue = load_pssm_file(pssm_path)
            result['pssm_issue'] = pssm_issue
            
            if pssm_content.size > 0:
                result['pssm_length'] = pssm_content.shape[0]
            
            if pssm_issue:
                stats['pssm_issues'] += 1
        
        # Check length match
        if result['dssp_length'] > 0 and result['fasta_length'] > 0:
            if result['dssp_length'] == result['fasta_length']:
                result['length_match'] = True
            else:
                stats['length_mismatches'] += 1
        
        # Add to results
        results.append(result)
        
        # Update statistics
        if (result['dssp_exists'] and not result['dssp_issue'] and 
            result['fasta_exists'] and not result['fasta_issue'] and 
            result['length_match']):
            stats['valid_files'] += 1
            stats['sequence_lengths'].append(result['dssp_length'])
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Valid files: {stats['valid_files']} ({stats['valid_files']/stats['total_files']*100:.2f}%)")
    print(f"DSSP issues: {stats['dssp_issues']} ({stats['dssp_issues']/stats['total_files']*100:.2f}%)")
    print(f"FASTA issues: {stats['fasta_issues']} ({stats['fasta_issues']/stats['total_files']*100:.2f}%)")
    print(f"PSSM issues: {stats['pssm_issues']} ({stats['pssm_issues']/stats['total_files']*100:.2f}%)")
    print(f"Length mismatches: {stats['length_mismatches']} ({stats['length_mismatches']/stats['total_files']*100:.2f}%)")
    
    if stats['sequence_lengths']:
        print("\nSequence Length Statistics:")
        print(f"Min length: {min(stats['sequence_lengths'])}")
        print(f"Max length: {max(stats['sequence_lengths'])}")
        print(f"Mean length: {np.mean(stats['sequence_lengths']):.2f}")
        print(f"Median length: {np.median(stats['sequence_lengths']):.2f}")
        
        # Plot sequence length distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(stats['sequence_lengths'], bins=50)
        plt.xlabel('Sequence Length')
        plt.ylabel('Count')
        plt.title('Sequence Length Distribution')
        
        # Save plot if output file is specified
        if output_file:
            plot_path = os.path.splitext(output_file)[0] + '_length_dist.png'
            plt.savefig(plot_path)
            print(f"Sequence length distribution plot saved to {plot_path}")
        
        plt.show()
    
    # Save results to CSV
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Check dataset for protein secondary structure prediction')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='dataset_check_results.csv', help='Path to save results (CSV)')
    args = parser.parse_args()
    
    check_dataset(args.data_dir, args.output)

if __name__ == '__main__':
    main() 