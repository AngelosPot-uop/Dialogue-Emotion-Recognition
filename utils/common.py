import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import torch.nn.functional as F

import config


import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(model, data_loader, criterion, debug=False):
    model.eval()
    model.to(config.device)
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_probs = []
    num_classes = get_num_classes_from_loader(data_loader)


    precision_metric = MulticlassPrecision(num_classes=num_classes, average='weighted').to(config.device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average='weighted').to(config.device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='weighted').to(config.device)
    num_classes = num_classes
    conf_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(config.device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            features, lengths, labels, mask = batch
            features, labels = features.to(config.device), labels.to(config.device)

            if debug:
                print(f"Batch {batch_idx + 1}/{len(data_loader)}")
                print(f"Features shape: {features.shape}")
                print(f"Lengths shape: {lengths.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Mask shape: {mask.shape}")

            outputs = model(features, lengths)

            if debug:
                print(f"Outputs shape: {outputs.shape}")

            active_outputs = []
            active_labels = []

            for i in range(outputs.shape[0]):  # iterate over the batch
                valid_length = lengths[i]
                active_outputs.append(outputs[i, :valid_length])
                active_labels.append(labels[i, :valid_length])

            active_outputs = torch.cat(active_outputs, dim=0)
            active_labels = torch.cat(active_labels, dim=0)

            if debug:
                print(f"Active outputs shape after concatenation: {active_outputs.shape}")
                print(f"Active labels shape after concatenation: {active_labels.shape}")

            active_outputs = active_outputs.view(-1, active_outputs.shape[-1])
            active_labels = active_labels.view(-1, active_labels.shape[-1])

            if debug:
                print(f"Flattened active outputs shape: {active_outputs.shape}")
                print(f"Flattened active labels shape: {active_labels.shape}")

            loss = criterion(active_outputs, active_labels.argmax(dim=1))
            total_loss += loss.item()

            preds = active_outputs.argmax(dim=1)
            target = active_labels.argmax(dim=1)

            precision_metric.update(preds, target)
            recall_metric.update(preds, target)
            f1_metric.update(preds, target)

            correct_predictions += torch.sum(preds == target)
            total_predictions += target.size(0)

            conf_matrix.update(preds, target)

            # Convert logits to probabilities using softmax
            probs = F.softmax(active_outputs, dim=1)

            # Collect all labels and probabilities
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()

    accuracy = correct_predictions.double() / total_predictions

    conf_matrix = conf_matrix.compute().cpu().numpy()
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # print("labels ", all_labels)
    # print("probs ", all_probs)

    return avg_loss, accuracy, conf_matrix, precision, recall, f1, all_probs, all_labels



def plot_confusion_matrix(conf_matrix, class_names, modelName):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {modelName}')
    plt.show()


def plot_precision_recall_curve(y_true, y_probs, num_classes, modelName):
    # Binarize the labels for multi-class precision-recall
    y_true_bin = label_binarize(y_true, classes=[i for i in range(len(num_classes))])

    plt.figure(figsize=(10, 7))
    for i in range(len(num_classes)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'{num_classes[i]}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{modelName} Precision-Recall Curve")
    plt.legend(loc="best")
    plt.show()


def plot_roc_curve(y_true, y_probs, num_classes, modelName):
    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=[i for i in range(len(num_classes))])

    plt.figure(figsize=(10, 7))
    for i in range(len(num_classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{num_classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({modelName})')
    plt.legend(loc="best")
    plt.show()

def plot_f1_scores(log_file, modelName):
    # Read the log file
    df = pd.read_csv(log_file, names=['Epoch', 'Train Loss', 'Val Loss', 'Val F1'])

    # Plot F1 score improvement across epochs
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Val F1'], marker='o')
    plt.xlabel('Epochs')
    plt.ylabel(f'Validation F1 Score')
    plt.title(f'F1 Score Improvement Across Epochs for {modelName}')
    plt.grid()
    plt.show()


def get_num_classes_from_loader(data_loader):
    # Access the dataset from the data_loader
    dataset = data_loader.dataset

    # Extract the labels from the first sample in the dataset
    first_sample = dataset[0]
    labels = first_sample[2]

    # Return the number of classes based on the labels shape
    return labels.shape[-1]


def save_results_to_csv(results, filename):
    results_df = pd.DataFrame(results)
    # Ensure that the tensor objects are converted to scalar values
    results_df['Test Accuracy'] = results_df['Test Accuracy'].apply(
        lambda x: x.item() if isinstance(x, torch.Tensor) else x)
    results_df['Precision'] = results_df['Precision'].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
    results_df['Recall'] = results_df['Recall'].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
    results_df['F1 Score'] = results_df['F1 Score'].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)

    # Remove empty or all-NA entries before concatenation
    results_df = results_df.dropna(how='all', axis=1)

    results_df.to_csv(filename, index=False)