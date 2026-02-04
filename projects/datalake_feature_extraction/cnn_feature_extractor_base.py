#!/usr/bin/env python3
"""
cnn_feature_extractor_base.py

Base module with shared CNN model architecture and utilities for feature extraction.
This module is imported by the narrative, ngram, and phrase extractors.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json


class EmbeddingDataset(Dataset):
    """Dataset for embedding sequences with MOE/MOP labels."""

    def __init__(
        self,
        embeddings: List[np.ndarray],
        labels: List[int],
        max_seq_len: int = None
    ):
        """
        Args:
            embeddings: List of embedding arrays, each shape (seq_len, embed_dim)
            labels: List of integer labels
            max_seq_len: Maximum sequence length (will pad/truncate)
        """
        self.embeddings = embeddings
        self.labels = labels

        # Determine embedding dimension from first non-empty embedding
        self.embed_dim = None
        for emb in embeddings:
            if emb is not None and len(emb) > 0:
                if isinstance(emb[0], (list, np.ndarray)):
                    self.embed_dim = len(emb[0])
                else:
                    self.embed_dim = len(emb)
                break

        if self.embed_dim is None:
            raise ValueError("Could not determine embedding dimension")

        # Determine max sequence length
        if max_seq_len is None:
            self.max_seq_len = max(
                len(emb) if emb is not None else 0
                for emb in embeddings
            )
        else:
            self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        label = self.labels[idx]

        # Convert to numpy array if needed
        if emb is None or len(emb) == 0:
            emb = np.zeros((self.max_seq_len, self.embed_dim))
        else:
            emb = np.array(emb)

        # Pad or truncate to max_seq_len
        if len(emb) < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - len(emb), self.embed_dim))
            emb = np.vstack([emb, padding])
        elif len(emb) > self.max_seq_len:
            emb = emb[:self.max_seq_len]

        # Shape: (seq_len, embed_dim) -> (1, seq_len, embed_dim) for CNN
        emb = emb.astype(np.float32)
        emb = np.expand_dims(emb, axis=0)  # Add channel dimension

        return torch.from_numpy(emb), torch.tensor(label, dtype=torch.long)


class CNNFeatureExtractor(nn.Module):
    """
    1D CNN for extracting features from embedding sequences.

    Architecture:
    - Multiple parallel conv layers with different kernel sizes
    - Max pooling over time
    - Fully connected layers for classification
    - Feature extraction from penultimate layer
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_filters: int = 128,
        kernel_sizes: List[int] = [3, 4, 5],
        dropout: float = 0.5,
        fc_hidden: int = 256
    ):
        super(CNNFeatureExtractor, self).__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        # Parallel conv layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(k, embed_dim),
                padding=(k // 2, 0)
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # Total features from all conv layers
        total_filters = num_filters * len(kernel_sizes)

        # Fully connected layers
        self.fc1 = nn.Linear(total_filters, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, 1, seq_len, embed_dim)

        Returns:
            logits: Classification logits (batch, num_classes)
        """
        # Apply each conv layer and max pool
        conv_outputs = []
        for conv in self.convs:
            # conv output: (batch, num_filters, seq_len, 1)
            c = self.relu(conv(x))
            # squeeze last dim: (batch, num_filters, seq_len)
            c = c.squeeze(3)
            # max pool over time: (batch, num_filters)
            c = torch.max(c, dim=2)[0]
            conv_outputs.append(c)

        # Concatenate all conv outputs: (batch, total_filters)
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def extract_features(self, x):
        """
        Extract features from penultimate layer (before final classification).

        Args:
            x: Input tensor of shape (batch, 1, seq_len, embed_dim)

        Returns:
            features: Feature tensor (batch, fc_hidden)
        """
        # Apply each conv layer and max pool
        conv_outputs = []
        for conv in self.convs:
            c = self.relu(conv(x))
            c = c.squeeze(3)
            c = torch.max(c, dim=2)[0]
            conv_outputs.append(c)

        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)

        # FC1 output is our feature vector
        features = self.relu(self.fc1(x))

        return features


def prepare_data_for_training(
    df: pd.DataFrame,
    embedding_col: str,
    tag_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List, List, List, List, LabelEncoder, List[int]]:
    """
    Prepare data for training by filtering labeled rows and splitting.

    Args:
        df: Input DataFrame
        embedding_col: Name of column containing embeddings
        tag_col: Name of column containing MOE/MOP tags
        test_size: Fraction for test split
        random_state: Random seed

    Returns:
        train_embeddings, train_labels, test_embeddings, test_labels, label_encoder, labeled_indices
    """
    # Filter rows that have valid tags (non-null, non-empty)
    labeled_mask = df[tag_col].notna() & (df[tag_col] != '')
    labeled_indices = df.index[labeled_mask].tolist()

    labeled_df = df.loc[labeled_mask].copy()

    if len(labeled_df) == 0:
        raise ValueError(f"No labeled data found in column '{tag_col}'")

    print(f"Found {len(labeled_df)} labeled rows out of {len(df)} total")

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labeled_df[tag_col].values)

    print(f"Classes: {label_encoder.classes_}")
    print(f"Class distribution:")
    for cls, count in zip(*np.unique(labels, return_counts=True)):
        print(f"  {label_encoder.inverse_transform([cls])[0]}: {count}")

    # Get embeddings
    embeddings = labeled_df[embedding_col].tolist()

    # Convert embeddings to proper format
    processed_embeddings = []
    for emb in embeddings:
        if isinstance(emb, str):
            # Parse JSON string if needed
            emb = json.loads(emb)
        if isinstance(emb, list):
            emb = np.array(emb)
        processed_embeddings.append(emb)

    # Split data
    train_emb, test_emb, train_labels, test_labels = train_test_split(
        processed_embeddings,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    return train_emb, train_labels, test_emb, test_labels, label_encoder, labeled_indices


def train_model(
    model: CNNFeatureExtractor,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping_patience: int = 10
) -> Dict[str, Any]:
    """
    Train the CNN model.

    Args:
        model: CNN model to train
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Maximum training epochs
        learning_rate: Learning rate
        device: Device to train on
        early_stopping_patience: Epochs without improvement before stopping

    Returns:
        Dictionary with training history and best model state
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    best_test_acc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_emb, batch_labels in train_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_emb)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_emb, batch_labels in test_loader:
                batch_emb = batch_emb.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_emb)
                loss = criterion(outputs, batch_labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = test_correct / test_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Early stopping check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history['best_test_acc'] = best_test_acc

    return history


def extract_features_for_all_rows(
    model: CNNFeatureExtractor,
    df: pd.DataFrame,
    embedding_col: str,
    max_seq_len: int,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> np.ndarray:
    """
    Extract features for all rows in the DataFrame (including unlabeled).

    Args:
        model: Trained CNN model
        df: Input DataFrame
        embedding_col: Name of embedding column
        max_seq_len: Maximum sequence length (from training)
        batch_size: Batch size for inference
        device: Device for inference

    Returns:
        Feature array of shape (num_rows, feature_dim)
    """
    model = model.to(device)
    model.eval()

    all_features = []

    # Process embeddings
    embeddings = df[embedding_col].tolist()

    # Determine embedding dimension
    embed_dim = None
    for emb in embeddings:
        if emb is not None:
            if isinstance(emb, str):
                emb = json.loads(emb)
            if len(emb) > 0:
                if isinstance(emb[0], (list, np.ndarray)):
                    embed_dim = len(emb[0])
                else:
                    embed_dim = len(emb)
                break

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch_embs = embeddings[i:i + batch_size]

            # Process batch
            batch_tensors = []
            for emb in batch_embs:
                if emb is None or (isinstance(emb, (list, np.ndarray)) and len(emb) == 0):
                    emb = np.zeros((max_seq_len, embed_dim))
                else:
                    if isinstance(emb, str):
                        emb = json.loads(emb)
                    emb = np.array(emb)

                # Pad or truncate
                if len(emb) < max_seq_len:
                    padding = np.zeros((max_seq_len - len(emb), embed_dim))
                    emb = np.vstack([emb, padding])
                elif len(emb) > max_seq_len:
                    emb = emb[:max_seq_len]

                emb = emb.astype(np.float32)
                emb = np.expand_dims(emb, axis=0)  # Add channel dim
                batch_tensors.append(emb)

            batch_tensor = torch.from_numpy(np.stack(batch_tensors)).to(device)
            features = model.extract_features(batch_tensor)
            all_features.append(features.cpu().numpy())

    return np.vstack(all_features)
