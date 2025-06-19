import argparse
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.model import FastRPModel
from src.data_loader import load_data, load_author_mappings

def main():
    parser = argparse.ArgumentParser(description="Predict co-authorship probability between two authors.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth file).')
    parser.add_argument('--author1', type=str, required=True, help='Name or ID of the first author.')
    parser.add_argument('--author2', type=str, required=True, help='Name or ID of the second author.')
    args = parser.parse_args()

    # --- Load Checkpoint and Data ---
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model_args = checkpoint['args']
    
    print("Loading data for model initialization...")
    # The model needs the 'relations' dictionary to be initialized, even if it's on CPU
    relations, n_authors, _, _ = load_data(model_args['data_dir'])
    
    print("Loading author mappings...")
    id_to_name, name_to_id = load_author_mappings(model_args['data_dir'])

    # --- Initialize Model ---
    print("Initializing model architecture...")
    model = FastRPModel(
        n_authors=n_authors,
        dim=model_args['dim'],
        meta_paths=model_args['meta_paths'],
        relations=relations, # Pass the loaded relations
        num_powers=model_args['num_powers'],
        alpha=model_args['alpha'],
        beta=model_args['beta'],
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- Find Author IDs ---
    def find_author_id(name_or_id):
        try:
            # First, try to interpret as a (0-based) integer ID
            return int(name_or_id)
        except ValueError:
            # If that fails, look up by name
            return name_to_id.get(name_or_id.lower())

    id1 = find_author_id(args.author1)
    id2 = find_author_id(args.author2)

    if id1 is None:
        print(f"Error: Author '{args.author1}' not found.")
        sys.exit(1)
    if id2 is None:
        print(f"Error: Author '{args.author2}' not found.")
        sys.exit(1)
        
    name1 = id_to_name.get(id1, f"ID:{id1}")
    name2 = id_to_name.get(id2, f"ID:{id2}")
    
    print(f"\\nFound authors:")
    print(f"  - '{name1}' (ID: {id1})")
    print(f"  - '{name2}' (ID: {id2})")

    # --- Predict Probability ---
    with torch.no_grad():
        id1_tensor = torch.tensor([id1])
        id2_tensor = torch.tensor([id2])
        probability = model(id1_tensor, id2_tensor).item()

    print(f"\\nPredicted probability of co-authorship: {probability:.2%}")

if __name__ == '__main__':
    main() 