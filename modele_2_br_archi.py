import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ECGConceptCNN(nn.Module):
    def __init__(self, concept_input_size, num_classes, ECG=True, Concept=True):
        super(ECGConceptCNN, self).__init__()

        self.use_ecg = ECG
        self.use_concept = Concept

        # === Branche ECG avec blocs résiduels ===
        if self.use_ecg:
            self.ecg_branch = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                ResidualBlock(32),

                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                ResidualBlock(64),

                nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2), # CORRECTION 1 
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                ResidualBlock(128),

                nn.Conv1d(128, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.AdaptiveAvgPool1d(1),  # Sortie : (batch_size, 256, 1)
                nn.Flatten(),              # → (batch_size, 256)

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, concept_input_size)
                # Pas de sigmoid ici car BCEWithLogitsLoss inclut la sigmoid
            )
            

        # === Branche Concept avec MLP amélioré ===
        if self.use_concept:
            self.concept_branch = nn.Sequential(
                nn.Linear(concept_input_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )

        # === Fusion et classification ===
        if self.use_ecg and self.use_concept:
            fusion_input_size = concept_input_size + 64
        elif self.use_ecg:
            fusion_input_size = concept_input_size
        elif self.use_concept:
            fusion_input_size = 64
        else:
            raise ValueError("Au moins une des branches ECG ou Concept doit être activée.")

        self.fc = nn.Sequential( # CORRECTION 2
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )


    def forward(self, ecg=None, concepts=None):
        features = []

        if self.use_ecg:
            if ecg is None:
                raise ValueError("Les données ECG sont requises car ECG=True")
            ecg_features = self.ecg_branch(ecg)
            features.append(ecg_features)

        if self.use_concept:
            if concepts is None:
                raise ValueError("Les données de concepts sont requises car Concept=True")
            concept_features = self.concept_branch(concepts)
            features.append(concept_features)

        if len(features) > 1:
            combined = torch.cat(features, dim=1)
        else:
            combined = features[0]

        return self.fc(combined)
