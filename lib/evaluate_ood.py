import numpy as np

import torch
import torch.nn.functional as F

import gpytorch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from .datasets import get_dataset


import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt


def prepare_ood_datasets(true_dataset, ood_dataset):
    ood_dataset.transform = true_dataset.transform

    datasets = [true_dataset, ood_dataset]

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    return dataloader, anomaly_targets


def loop_over_dataloader(model, likelihood, dataloader):
    model.eval()
    if likelihood is not None:
        likelihood.eval()

    with torch.no_grad():
        scores = []
        accuracies = []
        logits = []
        labels = []
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()

            if likelihood is None:
                logit = model(data)
                output = F.softmax(logit, dim=1)
                logits.append(logit)
                labels.append(target)
            else:
                with gpytorch.settings.num_likelihood_samples(32):
                    y_pred = model(data).to_data_independent_dist()
                    output = likelihood(y_pred).probs.mean(0)

            uncertainty = -(output * output.log()).sum(1)

            pred = torch.argmax(output, dim=1)
            accuracy = pred.eq(target)

            accuracies.append(accuracy.cpu().numpy())
            scores.append(uncertainty.cpu().numpy())
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies, logits, labels


def get_ood_metrics(in_dataset, out_dataset, model, likelihood=None, root="./"):
    _, _, _, in_dataset = get_dataset(in_dataset, root=root)
    _, _, _, out_dataset = get_dataset(out_dataset, root=root)

    dataloader, anomaly_targets = prepare_ood_datasets(in_dataset, out_dataset)

    scores, accuracies, logits, labels = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies[: len(in_dataset)])
    auroc = roc_auc_score(anomaly_targets, scores)
    ece = ECELoss()(logits, labels)

    precision, recall, _ = precision_recall_curve(anomaly_targets, scores)
    aupr = auc(recall, precision)

    return accuracy, ece, auroc, aupr


def get_ood_metrics_cifar10_c(in_dataset, out_dataset, model, likelihood=None, root="./", corruption_type="brightness", corruption_intensity=1):
    _, _, _, in_dataset = get_dataset(in_dataset, root=root)
    _, _, _, out_dataset = get_dataset(out_dataset, root=root, corruption_type=corruption_type, corruption_intensity=corruption_intensity)

    dataloader, anomaly_targets = prepare_ood_datasets(in_dataset, out_dataset)

    scores, accuracies, logits, labels = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies[: len(in_dataset)])
    auroc = roc_auc_score(anomaly_targets, scores)
    ece = ECELoss()(logits, labels)

    precision, recall, _ = precision_recall_curve(anomaly_targets, scores)
    aupr = auc(recall, precision)

    return accuracy, ece, auroc, aupr


def get_auroc_classification(data, model, likelihood=None):
    if isinstance(data, torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        dataloader = data

    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)

    return accuracy, roc_auc


# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    """
    Compute ECE (Expected Calibration Error)
    """

    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece