import torch
from torchmetrics.classification import MulticlassPrecision, MulticlassF1Score, MulticlassRecall

from utils.logger import log_metrics
from utils.common import evaluate
import config


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=5, device="cpu", debug=False):
    best_val_f1 = float('-inf')
    epochs_no_improve = 0
    model.to(config.device)

    precision_metric = MulticlassPrecision(num_classes=7, average='weighted').to(config.device)
    recall_metric = MulticlassRecall(num_classes=7, average='weighted').to(config.device)
    f1_metric = MulticlassF1Score(num_classes=7, average='weighted').to(config.device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            features, lengths, labels, mask = batch

            if debug:
                print(f"\nBatch {batch_idx+1}/{len(train_loader)}")
                print(f"Features shape: {features.shape} on {features.device}")
                print(f"Lengths shape: {lengths.shape} on {lengths.device}")
                print(f"Labels shape: {labels.shape} on {labels.device}")
                print(f"Mask shape: {mask.shape} on {mask.device}")

            features, labels, mask = features.to(config.device), labels.to(config.device), mask.to(config.device)

            optimizer.zero_grad()
            outputs = model(features, lengths)

            if debug:
                print(f"Outputs shape: {outputs.shape} on {outputs.device}")

            active_outputs = []
            active_labels = []

            for i in range(outputs.shape[0]):  # iterate over the batch
                valid_length = lengths[i].item()
                active_outputs.append(outputs[i, :valid_length])
                active_labels.append(labels[i, :valid_length])

            active_outputs = torch.cat(active_outputs, dim=0).to(config.device)
            active_labels = torch.cat(active_labels, dim=0).to(config.device)

            if debug:
                print(f"Active outputs shape after concatenation: {active_outputs.shape} on {active_outputs.device}")
                print(f"Active labels shape after concatenation: {active_labels.shape} on {active_labels.device}")

            active_outputs = active_outputs.view(-1, active_outputs.shape[-1]).to(config.device)
            active_labels = active_labels.view(-1, active_labels.shape[-1]).to(config.device)

            if debug:
                print(f"Flattened active outputs shape: {active_outputs.shape} on {active_outputs.device}")
                print(f"Flattened active labels shape: {active_labels.shape} on {active_labels.device}")

            loss = criterion(active_outputs, active_labels.argmax(dim=1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            preds = active_outputs.argmax(dim=1).to(config.device)
            target = active_labels.argmax(dim=1).to(config.device)

            if debug:
                print(f"Preds shape: {preds.shape} on {preds.device}")
                print(f"Target shape: {target.shape} on {target.device}")

            precision_metric.update(preds, target)
            recall_metric.update(preds, target)
            f1_metric.update(preds, target)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss, _, _, _, _, val_f1, _, _ = evaluate(model, val_loader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Val F1: {val_f1}')

        # Log the metrics
        log_metrics(epoch + 1, avg_train_loss, avg_val_loss, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), '../results/best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
