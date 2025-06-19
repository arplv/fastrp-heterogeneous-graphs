import argparse
import torch
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data_loader import load_author_mappings

def main():
    parser = argparse.ArgumentParser(description="Predict co-authorship probability using pre-computed embeddings.")
    parser.add_argument('--embeddings-path', type=str, default='author_embeddings.pt', help='Path to the saved author embeddings (.pt file).')
    parser.add_argument('--checkpoint-path', type=str, default='fastrp_model.pth', help='Path to the model checkpoint to load the intercept.')
    parser.add_argument('--data-dir', type=str, default='data', help='Path to the data directory to load author names.')
    parser.add_argument('author1', type=str, help='Name or ID of the first author.')
    parser.add_argument('author2', type=str, help='Name or ID of the second author.')
    args = parser.parse_args()

    # --- Load Pre-computed Data ---
    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings = torch.load(args.embeddings_path, map_location='cpu')
    
    print(f"Loading checkpoint from {args.checkpoint_path} to get model intercept...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    intercept = checkpoint.get('intercept', 0.0) # Default to 0.0 if not found
    
    print(f"Loading author mappings from {args.data_dir}...")
    id_to_name, name_to_id = load_author_mappings(args.data_dir)

    # --- Find Author IDs ---
    def find_author_id(name_or_id):
        try:
            return int(name_or_id)
        except ValueError:
            return name_to_id.get(name_or_id.lower())

    id1 = find_author_id(args.author1)
    id2 = find_author_id(args.author2)

    if id1 is None or id1 >= len(embeddings):
        print(f"Error: Author '{args.author1}' not found or ID out of bounds.")
        sys.exit(1)
    if id2 is None or id2 >= len(embeddings):
        print(f"Error: Author '{args.author2}' not found or ID out of bounds.")
        sys.exit(1)
        
    name1 = id_to_name.get(id1, f"ID:{id1}")
    name2 = id_to_name.get(id2, f"ID:{id2}")
    
    print(f"\\nFound authors:")
    print(f"  - '{name1}' (ID: {id1})")
    print(f"  - '{name2}' (ID: {id2})")

    # --- Predict Probability ---
    z1 = embeddings[id1]
    z2 = embeddings[id2]
    
    dist_sq = ((z1 - z2) ** 2).sum()
    logits = intercept - dist_sq
    probability = torch.sigmoid(logits).item()

    print(f"\\nPredicted probability of co-authorship: {probability:.2%}")

if __name__ == '__main__':
    main() 