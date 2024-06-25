from sklearn.model_selection import GroupKFold
import torch
from torch import nn
import numpy as np

import config
from models.StandardLSTM import StandardLSTM
from models.bcLSTM import BidirectionalLSTM
from models.bcLSTM_attention import BiLSTMWithAttention
from utils.train import train
from utils.common import evaluate, save_results_to_csv
from utils.data import create_tensor



def cross_validate(dataloader, args, num_folds=5):
    # Extract features, labels, lengths, masks, and dialogue IDs
    features = dataloader.train_dialogue_features
    labels = dataloader.train_dialogue_label
    lengths = dataloader.train_dialogue_length
    masks = dataloader.train_mask
    dialogue_ids = np.arange(len(features))  # Using indices as unique dialogue IDs


    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=num_folds)
    results = []

    # Iterate through each fold
    for fold, (train_indices, val_indices) in enumerate(gkf.split(features, labels, groups=dialogue_ids)):
        print(f"Fold {fold + 1} / {num_folds}")
        # print(f"Fold {fold + 1}", len(train_idx), len(val_idx))
        # return
        lengths = np.array(lengths)

        # Split data
        train_features, val_features = features[train_indices], features[val_indices]
        train_labels, val_labels = labels[train_indices], labels[val_indices]
        train_lengths, val_lengths = lengths[train_indices], lengths[val_indices]
        train_masks, val_masks = masks[train_indices], masks[val_indices]

        # Convert to PyTorch tensors
        train_loader = create_tensor(train_features, train_labels, train_lengths, train_masks)
        val_loader = create_tensor(val_features, val_labels, val_lengths, val_masks)

        # Initialize model
        input_dim = train_features.shape[2]
        output_dim = train_labels.shape[2]
        hidden_dim = 128
        num_layers = 3
        dropout_rate = 0.25

        # Inititalize model
        if args.model == 'lstm':
            model = StandardLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                                 dropout_rate=dropout_rate)
        elif args.model == 'bclstm':
            model = BidirectionalLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                                      dropout_rate=dropout_rate)
        elif args.model == 'bclstm2':
            model = BiLSTMWithAttention(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                                      dropout_rate=dropout_rate)

        model.to(config.device)
        # Define criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Train model
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=15, debug=False)

        # Evaluate model
        test_loader = create_tensor(dataloader.test_dialogue_features, dataloader.test_dialogue_label, dataloader.test_dialogue_length, dataloader.test_mask)
        test_loss, test_accuracy, confusion_matrix, precision, recall, f1, _, _ = evaluate(model, test_loader, criterion, debug=False)

        print(f'Fold {fold + 1} - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        print(f'Precision: {precision}, Recall: {recall}, F1 Score (weighted): {f1}')


        # Append results to DataFrame
        results.append(
            {'Fold': fold + 1, 'Test Loss': test_loss, 'Test Accuracy': test_accuracy, 'Precision': precision,
             'Recall': recall, 'F1 Score': f1}
        )
        # results_df = results_df._append({'Fold': fold + 1, 'Test Loss': test_loss, 'Test Accuracy': test_accuracy,'Precision': precision, 'Recall': recall, 'F1 Score': f1}, ignore_index=True)

    # Save DataFrame to a CSV file
    # results_df.to_csv(f'./results/{args.model}_cross_validation_results.csv', index=False)
    save_results_to_csv(results, f'./results/{args.model}_cross_validation_results.csv')
    # print(results_df)
    print(f'Results saved to ./results/{args.model}_cross_validation_results.csv')
    return
