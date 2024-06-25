# Data Loader class from MELD github
import torch
import torch.nn as nn
import torch.cuda
from torch.utils.data import DataLoader as DataLoader, TensorDataset

class StandardLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(StandardLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, lengths):
        # Pack the sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)

        # Apply the linear layer on the output of the last LSTM layer
        out = self.fc(output)
        return out


# Unused
def get_lstm_model(dataloader, hidden_dim=128, num_layers=3, dropout_rate=0.25):
    # Set hyperparameters
    input_dim = dataloader.train_dialogue_features.shape[2]  # Based on embeddings size
    output_dim = dataloader.train_dialogue_label.shape[2]  # Based on labels size

    # Initialize the model
    model = StandardLSTM(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
    print(model)

    return model
