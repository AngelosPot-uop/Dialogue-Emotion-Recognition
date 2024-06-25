import torch
from torch.utils.data import TensorDataset, DataLoader
import config

def create_tensor(features, labels, lengths, masks):
    # Create train tensor
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32).to(config.device),
                                  torch.tensor(lengths, dtype=torch.long),
                                  torch.tensor(labels, dtype=torch.long).to(config.device),
                                  torch.tensor(masks, dtype=torch.float32).to(config.device))

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    return data_loader