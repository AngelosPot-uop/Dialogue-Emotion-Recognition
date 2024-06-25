import argparse
import os

# Import the Data Loader class (source: MELD github)
from baseline.data_helpers import Dataloader as MELD_DataLoader

from utils.logger import configure_logger
from models.StandardLSTM import StandardLSTM
from models.bcLSTM import BidirectionalLSTM
from models.bcLSTM_attention import BiLSTMWithAttention
from utils.train import train
from utils.validation import cross_validate
from utils.common import evaluate, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, plot_f1_scores
import config

import torch
import torch.nn as nn
import torch.cuda
import numpy as np

from utils.data import create_tensor


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # Initialize the dataloader for a specific mode
    dataloader = MELD_DataLoader(mode=args.classify)  # Options: 'Sentiment' (3 classes), 'Emotion' (7 classes)

    # Load data from MELD helper class
    print(f"Modality: {args.modality}")
    if args.modality == 'text':
        dataloader.load_text_data()
    elif args.modality == 'audio':
        dataloader.load_audio_data()
    elif args.modality == 'bimodal':
        dataloader.load_bimodal_data()


    # Get dimensions of current modality
    input_dim = dataloader.train_dialogue_features.shape[2]  # Based on embeddings size
    output_dim = dataloader.train_dialogue_label.shape[2]  # Based on labels size

    if args.cross_validate:
        cross_validate(dataloader, args)
    else:
        # Inititalize model
        if args.model == 'lstm':
            model = StandardLSTM(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_layers=3,
                                 dropout_rate=0.25)
        elif args.model == 'bclstm':
            model = BidirectionalLSTM(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_layers=3,
                                      dropout_rate=0.25)
        elif args.model == 'bclstm2':
            model = BiLSTMWithAttention(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_layers=3,
                                      dropout_rate=0.25)
        # model.to(config.device)


        # Define Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        train_loader = create_tensor(dataloader.train_dialogue_features, dataloader.train_dialogue_label,
                                     dataloader.train_dialogue_length, dataloader.train_mask)
        val_loader = create_tensor(dataloader.val_dialogue_features, dataloader.val_dialogue_label,
                                   dataloader.val_dialogue_length, dataloader.val_mask)

        # Start training. Use 'debug' option to show the print statements on each batch.
        # train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=15,debug=True)

        # Clear the log file and configure the logger
        log_file = '../results/training.log'
        open(log_file, 'w').close()
        configure_logger(log_file)


        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=2, patience=15, debug=False)
        plot_f1_scores('../results/training.log', args.model)

        test_loader = create_tensor(dataloader.test_dialogue_features, dataloader.test_dialogue_label, dataloader.test_dialogue_length, dataloader.test_mask)



        # Evaluate the model
        test_loss, test_accuracy, confusion_matrix, precision, recall, f1, all_probs, all_labels = evaluate(model, test_loader, criterion)
        print("class names are: ", dataloader.class_names)
        plot_confusion_matrix(confusion_matrix, dataloader.class_names, args.model)
        # # test_loss, test_accuracy, confusion_matrix, precision, recall, f1 = evaluate(model, test_loader, criterion)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        print(f'Precision: {precision}, Recall: {recall}, Weighted F1 Score: {f1}')
        # print(f'Confusion Matrix:\n{confusion_matrix}')

        plot_precision_recall_curve(all_labels, all_probs, num_classes=dataloader.class_names, modelName=args.model)
        plot_roc_curve(all_labels, all_probs, num_classes=dataloader.class_names, modelName=args.model)


if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.required = True
    parser.add_argument("--classify", help="Options: 'Emotion' or 'Sentiment'.", required=True)
    parser.add_argument("--modality", help="Options: 'text', 'audio', 'bimodal'.", required=True)
    parser.add_argument("--model", help="Options: 'lstm', 'bclstm'.", required=True)
    parser.add_argument("--cross_validate", default=False, action="store_true")
    args = parser.parse_args()

    args.classify = args.classify.title()
    args.modality = args.modality.lower()
    args.modality = args.modality.lower()

    if args.classify not in ["Emotion", "Sentiment"]:
        raise ValueError("Invalid --classify flag. Valid options are: 'Emotion', 'Sentiment'.")
        exit()
    if args.modality not in ["text", "audio", "bimodal"]:
        raise ValueError("Invalid --modality flag. Valid options are: 'text', 'audio', 'bimodal'.")
        exit()
    if args.model not in ["lstm", "bclstm", "bclstm2"]:
        raise ValueError("Invalid --model flag. Valid options are: 'lstm', 'bclstm', 'bclstm2'.")
        exit()


    results_dir= "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device is: ", config.device, ". GPU available? ", torch.cuda.is_available())


    # Set seeds for reproducible results
    set_seed(1234)

    # Begin the process
    main(args)
