import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

from src.data_loader import load_author_mappings, load_dictionary
from src.model import FastRPModel # Needed to load the checkpoint

def main(args):
    """Predicts the link probability between two nodes in the joint embedding space."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Model and Embeddings ---
    print(f"Loading model checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']
    
    # We need to instantiate the model to get the final parameters (slope, intercept)
    # The feature matrices themselves are not needed for prediction, so we can pass dummy values.
    # NOTE: This is a bit of a hack. A better approach might be to save slope/intercept separately.
    n_total = 1 # Dummy value
    dummy_relations = {} # Dummy value
    model = FastRPModel(
        n_total=n_total,
        dim=model_args['dim'],
        meta_paths=model_args['meta_paths'],
        relations=dummy_relations,
        num_powers=model_args['num_powers'],
        alpha=model_args['alpha'],
        beta=model_args['beta'],
        s=model_args['s'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded.")

    print(f"Loading joint embeddings from {args.embeddings_path}...")
    embeddings_by_type = torch.load(args.embeddings_path, map_location=device)
    print("Embeddings loaded.")

    # --- 2. Load Dictionaries for ID Lookup ---
    print("Loading dictionaries...")
    author_id_to_name, author_name_to_id = load_author_mappings(args.data_dir)
    conf_name_to_id = load_dictionary(Path(args.data_dir) / 'conf_dict.txt')
    term_name_to_id = load_dictionary(Path(args.data_dir) / 'term_dict.txt')
    
    name_to_id_map = {'A': author_name_to_id, 'C': conf_name_to_id, 'T': term_name_to_id}
    id_to_name_map = {'A': author_id_to_name, 'C': {v: k for k, v in conf_name_to_id.items()}, 'T': {v: k for k, v in term_name_to_id.items()}}

    # --- 3. Get Embeddings for Specified Nodes ---
    try:
        # Node 1
        node1_name_lower = args.node1_name.lower()
        node1_id = name_to_id_map[args.node1_type].get(node1_name_lower)
        if node1_id is None:
            raise ValueError(f"Node 1 '{args.node1_name}' of type '{args.node1_type}' not found.")
        emb1 = embeddings_by_type[args.node1_type][node1_id]
        node1_display_name = id_to_name_map[args.node1_type].get(node1_id, args.node1_name)

        # Node 2
        node2_name_lower = args.node2_name.lower()
        node2_id = name_to_id_map[args.node2_type].get(node2_name_lower)
        if node2_id is None:
            raise ValueError(f"Node 2 '{args.node2_name}' of type '{args.node2_type}' not found.")
        emb2 = embeddings_by_type[args.node2_type][node2_id]
        node2_display_name = id_to_name_map[args.node2_type].get(node2_id, args.node2_name)

    except KeyError as e:
        print(f"Error: Invalid node type '{e.args[0]}'. Please use 'A', 'C', or 'T'.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- 4. Calculate Link Probability ---
    with torch.no_grad():
        dist_sq = ((emb1 - emb2) ** 2).sum()
        logits = model.intercept - model.slope * dist_sq
        probability = torch.sigmoid(logits).item()

    print("\\n--- Link Prediction ---")
    print(f"Node 1: [{args.node1_type}] {node1_display_name} (ID: {node1_id})")
    print(f"Node 2: [{args.node2_type}] {node2_display_name} (ID: {node2_id})")
    print(f"Latent Distance Squared: {dist_sq:.4f}")
    print(f"Logit: {logits:.4f}")
    print(f"Predicted Link Probability: {probability:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict link probability between two nodes using a trained FastRP model.")
    parser.add_argument('--model-path', type=Path, default='fastrp_model.pth', help='Path to the trained model checkpoint.')
    parser.add_argument('--embeddings-path', type=Path, default='joint_embeddings.pt', help='Path to the saved joint embeddings file.')
    parser.add_argument('--data-dir', type=Path, default='data', help='Directory containing the dataset dictionaries.')
    
    parser.add_argument('--node1-type', type=str, required=True, choices=['A', 'C', 'T'], help="Type of the first node ('A' for Author, 'C' for Conference, 'T' for Term).")
    parser.add_argument('--node1-name', type=str, required=True, help="Name of the first node.")
    
    parser.add_argument('--node2-type', type=str, required=True, choices=['A', 'C', 'T'], help="Type of the second node.")
    parser.add_argument('--node2-name', type=str, required=True, help="Name of the second node.")

    main(parser.parse_args()) 