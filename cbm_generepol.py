# import des packages utiles
import os
import csv

import neurokit2 as nk

import numpy as np

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import wfdb
from wfdb import processing


from tqdm import tqdm 

import matplotlib.pyplot as plt

from ConceptsECGanalys import *

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ECGConceptCBM(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super(ECGConceptCBM, self).__init__()

        # Extraction des concepts depuis les ECG avec blocs résiduels
        self.ecg_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            ResidualBlock(32),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            ResidualBlock(64),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            ResidualBlock(128),

            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_concepts),
            # Pas de sigmoid ici car BCEWithLogitsLoss inclut la sigmoid
        )

        # Prédiction finale basée sur les concepts
        self.fc = nn.Sequential(
            nn.Linear(num_concepts, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, ecg):
        pred_concepts = self.ecg_branch(ecg)
        output = self.fc(pred_concepts)
        return pred_concepts, output


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, Lambda=0.1,
                device="cuda", display=10, early_stopping_patience=10,
                save_path="models/cbm/best_models.pt", lr_dyn=False):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    loss_task = nn.CrossEntropyLoss()
    loss_concept = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")

    # Tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_task_losses, val_task_losses = [], []
    train_concept_losses, val_concept_losses = [], []
    learning_rates = []

    loss_train_fin, acc_train_fin, loss_val_fin, acc_val_fin, ep = 0, 0, 0, 0, 0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct_preds, total_samples = 0.0, 0, 0
        total_task_loss, total_concept_loss = 0.0, 0.0

        for ecg_data, concept_labels, class_labels in train_loader:
            ecg_data, concept_labels, class_labels = ecg_data.to(device), concept_labels.to(device), class_labels.to(device)

            optimizer.zero_grad()
            pred_concepts, pred_class = model(ecg_data)

            task_loss = loss_task(pred_class, class_labels)
            concept_loss = loss_concept(pred_concepts, concept_labels)
            loss = task_loss + Lambda * concept_loss

            loss.backward()
            optimizer.step()

            batch_size = class_labels.size(0)
            total_loss += loss.item() * batch_size
            total_task_loss += task_loss.item() * batch_size
            total_concept_loss += concept_loss.item() * batch_size

            _, predicted = torch.max(pred_class, 1)
            correct_preds += (predicted == class_labels).sum().item()
            total_samples += batch_size

        train_acc = correct_preds / total_samples
        train_losses.append(total_loss / total_samples)
        train_accuracies.append(train_acc)
        train_task_losses.append(total_task_loss / total_samples)
        train_concept_losses.append(total_concept_loss / total_samples)

        # Validation
        model.eval()
        val_loss, val_task_loss, val_concept_loss = 0.0, 0.0, 0.0
        val_correct, val_samples = 0, 0

        with torch.no_grad():
            for ecg_data, concept_labels, class_labels in val_loader:
                ecg_data, concept_labels, class_labels = ecg_data.to(device), concept_labels.to(device), class_labels.to(device)
                pred_concepts, pred_class = model(ecg_data)

                task_loss = loss_task(pred_class, class_labels)
                concept_loss = loss_concept(pred_concepts, concept_labels)
                loss = task_loss + Lambda * concept_loss

                batch_size = class_labels.size(0)
                val_loss += loss.item() * batch_size
                val_task_loss += task_loss.item() * batch_size
                val_concept_loss += concept_loss.item() * batch_size

                _, predicted = torch.max(pred_class, 1)
                val_correct += (predicted == class_labels).sum().item()
                val_samples += batch_size

        val_acc = val_correct / val_samples
        val_losses.append(val_loss / val_samples)
        val_accuracies.append(val_acc)
        val_task_losses.append(val_task_loss / val_samples)
        val_concept_losses.append(val_concept_loss / val_samples)

        # Log learning rate
        if lr_dyn:
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

        # Scheduler step
        scheduler.step(val_losses[-1])

        if epoch % display == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_losses[-1]:.4f}, Acc: {val_acc:.4f}")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
            loss_train_fin, acc_train_fin = train_losses[-1], train_acc
            loss_val_fin, acc_val_fin, ep = val_losses[-1], val_acc, epoch
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print("Entraînement terminé")
    print(f"Epoch {ep+1}/{num_epochs} | Train Loss: {loss_train_fin:.4f}, Acc: {acc_train_fin:.4f} | Val Loss: {loss_val_fin:.4f}, Acc: {acc_val_fin:.4f}")

    if val_loader is not None:
        model.load_state_dict(torch.load(save_path))
        print("Meilleur modèle chargé depuis sauvegarde.")

    if lr_dyn:
        return (train_losses, val_losses, train_accuracies, val_accuracies,
                train_task_losses, val_task_losses, train_concept_losses, val_concept_losses,
                learning_rates)
    else:
        return (train_losses, val_losses, train_accuracies, val_accuracies,
                train_task_losses, val_task_losses, train_concept_losses, val_concept_losses)

def evaluate_model(model, data_loader, Lambda=0.1, device="cuda"):
    """
    Évalue le modèle ECGConceptCBM sur un ensemble de test/validation.

    :param model: Modèle ECGConceptCBM entraîné
    :param data_loader: Dataloader pour l'évaluation
    :param λ: Poids de la perte des concepts supervisés
    :param device: "cuda" ou "cpu"
    :return: (avg_loss, accuracy, all_labels, all_predictions, all_probs)
    """
    model.to(device)
    model.eval()

    loss_task = nn.CrossEntropyLoss()
    loss_concept = nn.BCEWithLogitsLoss()

    total_loss, correct_preds, total_samples = 0.0, 0, 0
    all_labels = []
    all_predictions = []
    all_probs = []  # probabilités pour la classe positive uniquement (classe 1)

    with torch.no_grad():
        for ecg_data, concept_labels, class_labels, _ in data_loader:
            ecg_data = ecg_data.to(device)
            concept_labels = concept_labels.to(device)
            class_labels = class_labels.to(device)

            pred_concepts, pred_class = model(ecg_data)

            # loss 
            task_loss = loss_task(pred_class, class_labels)
            concept_loss = loss_concept(pred_concepts, concept_labels)
            loss = task_loss + Lambda * concept_loss
            total_loss += loss.item()

            # accuracy et predictions
            probs = torch.softmax(pred_class, dim=1)
            _, predicted = torch.max(probs, 1)

            all_labels.extend(class_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # probabilité classe 1

            correct_preds += (predicted == class_labels).sum().item()
            total_samples += class_labels.size(0)

    avg_test_loss = total_loss / len(data_loader)
    test_accuracy = correct_preds / total_samples

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    return avg_test_loss, test_accuracy, all_labels, all_predictions, all_probs

# concepts continus
class ECGConceptDataset(Dataset):
    def __init__(self, csv_path, concepts_csv_path, split_path=None):
        ecg_df = pd.read_csv(csv_path)

        # appliquer le split si précisé
        if split_path is not None:
            import h5py
            with h5py.File(split_path, "r") as f:
                split_ids = {f["uid"][i].decode("utf-8") for i in range(f["uid"].shape[0])}
            ecg_df = ecg_df[ecg_df["ECG-id"].isin(split_ids)]

        # identifier les colonnes contenant le signal ECG
        self.signal_columns = [col for col in ecg_df.columns if col not in ['ECG-id', 'class']]
        self.signals = ecg_df[self.signal_columns].values.astype(np.float32)

        # normalisation des signaux ECG (z-score)
        self.signals = (self.signals - np.mean(self.signals, axis=1, keepdims=True)) / \
                       (np.std(self.signals, axis=1, keepdims=True) + 1e-8)

        # stocker les ECG-id et labels
        self.ecg_ids = ecg_df['ECG-id'].values
        ecg_df['class'] = ecg_df['class'].apply(lambda x: 1 if str(x).lower() == 'sotalol' else 0)
        self.labels = ecg_df['class'].values.astype(np.int64)

        # chargement des concepts et normalisation Min-Max
        self.concept_features = [
            'p-duration', 'pp-interval', 'pr-interval', 'pr-segment', 'qrs-duration',
            'qt-duration', 'rr-interval', 'st-segment', 'stt-segment', 't-duration',
            'tp-interval'
        ]
        self.concepts_df = self._load_concepts(concepts_csv_path, self.concept_features)
        self.available_concepts_ids = set(self.concepts_df.index)

    def _load_concepts(self, path, concept_features):
        df = pd.read_csv(path, header=[0, 1])
        df.columns = pd.MultiIndex.from_tuples(
            [('ECG-id', '')] + [(c1.strip(), c2.strip()) for c1, c2 in df.columns[1:]]
        )
        df.set_index(('ECG-id', ''), inplace=True)

        selected_columns = []
        for concept in concept_features:
            selected_columns.append((concept, 'mean'))
            selected_columns.append((concept, 'std'))

        concept_df = df[selected_columns].copy()
        concept_df.columns = [f"{c1}-{c2}" for c1, c2 in concept_df.columns]

        # normalisation Min-Max
        concept_df = (concept_df - concept_df.min()) / (concept_df.max() - concept_df.min() + 1e-8)

        return concept_df

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)  # (1, signal_length)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        ecg_id = self.ecg_ids[idx]

        valid_flag = ecg_id in self.available_concepts_ids
        if valid_flag:
            concept_values = self.concepts_df.loc[ecg_id].fillna(0).values.astype(np.float32)
        else:
            concept_values = np.zeros(len(self.concepts_df.columns), dtype=np.float32)

        concept_tensor = torch.tensor(concept_values, dtype=torch.float32)

        return signal, concept_tensor, label


# concepts binaires

class ECGConceptDatasetBinaire(Dataset):
    def __init__(self, csv_path, concepts_csv_path, split_path=None):
        ecg_df = pd.read_csv(csv_path)

        if split_path is not None:
            import h5py
            with h5py.File(split_path, "r") as f:
                split_ids = {f["uid"][i].decode("utf-8") for i in range(f["uid"].shape[0])}
            ecg_df = ecg_df[ecg_df["ECG-id"].isin(split_ids)]

        # Identification des colonnes de signal ECG
        self.signal_columns = [col for col in ecg_df.columns if col not in ['ECG-id', 'class']]
        self.signals = ecg_df[self.signal_columns].values.astype(np.float32)

        # Normalisation des signaux ECG (z-score)
        self.signals = (self.signals - np.mean(self.signals, axis=1, keepdims=True)) / \
                       (np.std(self.signals, axis=1, keepdims=True) + 1e-8)

        # Stocker les ECG-id et labels
        self.ecg_ids = ecg_df['ECG-id'].values
        ecg_df['class'] = ecg_df['class'].apply(lambda x: 1 if str(x).lower() == 'sotalol' else 0)
        self.labels = ecg_df['class'].values.astype(np.int64)

        # Chargement des concepts
        self.concept_features = [
            'p-duration', 'pp-interval', 'pr-interval', 'pr-segment', 'qrs-duration',
            'qt-duration', 'rr-interval', 'st-segment', 'stt-segment', 't-duration',
            'tp-interval'
        ]
        self.concepts_df = self._load_concepts(concepts_csv_path, self.concept_features)

        # Sauvegarde des IDs disponibles pour les concepts
        self.available_concepts_ids = set(self.concepts_df.index)

    def _load_concepts(self, path, concept_features):
        df = pd.read_csv(path, header=[0, 1])

        # Nettoyer les colonnes multi-index
        df.columns = pd.MultiIndex.from_tuples([(c1.strip(), c2.strip()) for c1, c2 in df.columns])

        # Identifier la colonne 'ECG-id'
        id_col = [col for col in df.columns if col[0] == 'ECG-id']
        if not id_col:
            raise KeyError("Impossible de trouver la colonne 'ECG-id' dans les niveaux de colonnes.")
        df.set_index(id_col[0], inplace=True)

        # Sélectionner les colonnes des concepts (long + short)
        selected_columns = [
            (c1, metric)
            for c1 in concept_features
            for metric in ['long', 'short']
            if (c1, metric) in df.columns
        ]

        if not selected_columns:
            raise ValueError(f"Aucune colonne valide trouvée pour les concepts : {concept_features}")

        concept_df = df[selected_columns].copy()
        concept_df.columns = [f"{c1}-{c2}" for c1, c2 in concept_df.columns]

        # Vérifier si les données sont binaires
        is_binary = concept_df.dropna().isin([0, 1]).all().all()

        if is_binary:
            print("Colonnes déjà binaires – pas de binarisation appliquée.")
            concept_df = concept_df.fillna(0).astype(np.float32)
        else:
            print("Données non binaires – binarisation par médiane appliquée.")
            threshold = concept_df.median()
            concept_df = (concept_df >= threshold).astype(np.float32)

        return concept_df

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)  # (1, signal_length)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        ecg_id = self.ecg_ids[idx]

        valid_flag = ecg_id in self.available_concepts_ids
        if valid_flag:
            concept_values = self.concepts_df.loc[ecg_id].fillna(0).values.astype(np.float32)
        else:
            concept_values = np.zeros(len(self.concept_features) * 2, dtype=np.float32)  # x2: long + short

        concept_tensor = torch.tensor(concept_values, dtype=torch.float32)

        return signal, concept_tensor, label

