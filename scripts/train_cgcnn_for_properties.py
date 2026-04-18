"""
Train CGCNN model for formation energy and band gap prediction on MP-20 dataset.

This script trains two separate CGCNN models:
1. Formation energy prediction
2. Band gap prediction

Requirements:
    pip install torch-geometric pymatgen
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice


class SimpleCGCNN(nn.Module):
    """
    Simplified CGCNN for property prediction.
    """
    def __init__(self, atom_fea_len=64, h_fea_len=128, n_conv=3, n_h=1):
        super(SimpleCGCNN, self).__init__()

        # Atom embedding
        self.embedding = nn.Linear(92, atom_fea_len)  # 92 elements

        # Convolution layers
        self.convs = nn.ModuleList([
            CGConv(atom_fea_len, dim=1, batch_norm=True)
            for _ in range(n_conv)
        ])

        # Fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(atom_fea_len, h_fea_len)
        ])
        for _ in range(n_h - 1):
            self.fc_layers.append(nn.Linear(h_fea_len, h_fea_len))

        # Output layer
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Embedding
        x = self.embedding(x)

        # Convolution
        for conv in self.convs:
            x = F.softplus(conv(x, edge_index, edge_attr))

        # Pooling
        x = global_mean_pool(x, batch)

        # Fully connected
        for fc in self.fc_layers:
            x = F.softplus(fc(x))

        # Output
        out = self.fc_out(x)

        return out.squeeze(-1)


def structure_to_graph(structure, radius=8.0):
    """
    Convert pymatgen Structure to PyG Data object.

    Args:
        structure: pymatgen Structure
        radius: cutoff radius for neighbors

    Returns:
        PyG Data object
    """
    # Get atom features (one-hot encoding of atomic numbers)
    atom_types = [site.specie.Z for site in structure]
    x = torch.zeros(len(atom_types), 92)
    for i, z in enumerate(atom_types):
        x[i, z - 1] = 1.0

    # Get edges (neighbors within radius)
    neighbors = structure.get_all_neighbors(radius, include_index=True)
    edge_index = []
    edge_attr = []

    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            j = neighbor.index
            distance = neighbor.nn_distance
            edge_index.append([i, j])
            edge_attr.append([distance])

    if len(edge_index) == 0:
        # No neighbors found, add self-loops
        edge_index = [[i, i] for i in range(len(atom_types))]
        edge_attr = [[0.0] for _ in range(len(atom_types))]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def load_dataset(csv_path, target_property='formation_energy_per_atom', radius=8.0):
    """
    Load dataset from CSV and convert to PyG format.

    Args:
        csv_path: Path to CSV file
        target_property: Property to predict ('formation_energy_per_atom' or 'band_gap')
        radius: Cutoff radius for neighbors

    Returns:
        List of PyG Data objects
    """
    df = pd.read_csv(csv_path)

    data_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {csv_path}"):
        try:
            # Parse CIF
            structure = Structure.from_str(row['cif'], fmt='cif')

            # Convert to graph
            data = structure_to_graph(structure, radius=radius)

            # Add target
            data.y = torch.tensor([row[target_property]], dtype=torch.float)

            # Add material_id for reference
            data.material_id = row['material_id']

            data_list.append(data)

        except Exception as e:
            print(f"Error processing {row.get('material_id', idx)}: {e}")
            continue

    return data_list


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)

            total_loss += loss.item() * data.num_graphs
            predictions.extend(out.cpu().numpy())
            targets.extend(data.y.cpu().numpy())

    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))

    return total_loss / len(loader.dataset), mae


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    print(f"\nLoading datasets for {args.target_property}...")
    train_data = load_dataset(args.train_csv, target_property=args.target_property, radius=args.radius)
    val_data = load_dataset(args.val_csv, target_property=args.target_property, radius=args.radius)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create model
    model = SimpleCGCNN(
        atom_fea_len=args.atom_fea_len,
        h_fea_len=args.h_fea_len,
        n_conv=args.n_conv,
        n_h=args.n_h
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    # Training loop
    best_val_mae = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_path = Path(args.output_dir) / f"cgcnn_{args.target_property}_best.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'args': vars(args),
            }, save_path)

            print(f"Saved best model to {save_path}")

    print(f"\nTraining completed. Best Val MAE: {best_val_mae:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CGCNN for property prediction')

    # Data
    parser.add_argument('--train-csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val-csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--target-property', type=str, required=True,
                        choices=['formation_energy_per_atom', 'band_gap'],
                        help='Property to predict')
    parser.add_argument('--radius', type=float, default=8.0, help='Cutoff radius for neighbors')

    # Model
    parser.add_argument('--atom-fea-len', type=int, default=64, help='Atom feature length')
    parser.add_argument('--h-fea-len', type=int, default=128, help='Hidden feature length')
    parser.add_argument('--n-conv', type=int, default=3, help='Number of convolution layers')
    parser.add_argument('--n-h', type=int, default=1, help='Number of hidden layers')

    # Training
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')

    # Output
    parser.add_argument('--output-dir', type=str, default='./cgcnn_models', help='Output directory')

    args = parser.parse_args()
    main(args)
