import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, MoleculeNet
from torch_geometric.data import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from torch_geometric.datasets import BAMultiShapesDataset


# ---------------------------
# Transform: degree features
# ---------------------------
class DegreeFeatures(BaseTransform):
    """Transform that uses node degree as integer features"""
    def __call__(self, data):
        row, col = data.edge_index
        deg = degree(col, num_nodes=data.num_nodes, dtype=torch.long)
        data.x = deg.float().unsqueeze(1)  # Shape: [num_nodes, 1]
        return data


# ---------------------------
# Atom dictionary & utils
# ---------------------------
original_atom_dict = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 15: "P", 16: "S", 17: "Cl", 20: "Ca", 35: "Br", 53: "I"
}
atom_types = sorted(original_atom_dict.keys())
atom_to_idx = {atom_num: idx for idx, atom_num in enumerate(atom_types)}
num_atom_types = len(atom_types)
atom_type_dict = {idx: original_atom_dict[atom_types[idx]] for idx in range(num_atom_types)}
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def convert_atoms_to_onehot(x, atom_to_idx, num_atom_types):
    """Convert atomic numbers to one-hot encodings - ONLY use first column (atomic numbers)"""
    atomic_numbers = x[:, 0].long()
    atom_indices = torch.tensor([atom_to_idx.get(atom_num.item(), -1) for atom_num in atomic_numbers])

    valid_mask = atom_indices >= 0
    one_hot = torch.zeros(len(atomic_numbers), num_atom_types, dtype=torch.float)
    if valid_mask.any():
        one_hot[valid_mask] = F.one_hot(atom_indices[valid_mask], num_classes=num_atom_types).float()
    return one_hot


# ---------------------------
# Data loader function
# ---------------------------
def load_data(name, seed=42):
    torch.manual_seed(seed)
    print("Seed:", seed)
    if name == "IMDB-BINARY":
        dataset = TUDataset(root='./data', name='IMDB-BINARY', transform=DegreeFeatures())
    elif name=="reddit_threads":
        dataset = TUDataset(root='./data', name='reddit_threads', transform=DegreeFeatures())
    elif name=="twitch_egos":
        dataset = TUDataset(root='./data', name='twitch_egos', transform=DegreeFeatures())
    elif name=="github_stargazers":
        dataset = TUDataset(root='./data', name='github_stargazers', transform=DegreeFeatures())
    elif name == "BBBP":
        dataset = MoleculeNet(root='./data', name='BBBP')
    elif name == "Mutagenicity":
        dataset = TUDataset(root='./data', name="Mutagenicity")
    elif name == "NCI1":
        dataset = TUDataset(root='./data', name='NCI1')
    elif name == "BAMultiShapes":
        dataset = BAMultiShapesDataset(root='./data')
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Shuffle & split
    dataset = dataset.shuffle()
    split_point = int(len(dataset) * 0.8)
    train_dataset = [dataset[i].clone() for i in range(split_point)]
    test_dataset = [dataset[i].clone() for i in range(split_point, len(dataset))]

    # Special preprocessing for BBBP
    if name == "BBBP":
        for data in train_dataset + test_dataset:
            if data.y.dim() > 1:
                data.y = data.y.squeeze(-1)
            data.y = data.y.long()
            data.x = convert_atoms_to_onehot(data.x, atom_to_idx, num_atom_types)
    if name in ["reddit_threads", "twitch_egos", "github_stargazers"]:
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader, device
