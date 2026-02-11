"""
CNN Classifier Module
=====================

1D Convolutional Neural Network for text classification and feature
importance extraction from vectorized text representations.

Architecture:
    Conv1d(embedding_dim, 100, kernel_size=3) -> ReLU -> AdaptiveMaxPool1d -> Linear(100, num_classes)

The feature importance extraction uses gradient-based attribution:
backpropagating from the predicted class through the input to compute
token-level importance scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional


def prepare_data(tokens, vectors, labels, max_len: int = 100, 
                 vector_dim: int = 100) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Prepare data for CNN training.
    
    Pads or truncates vector sequences to uniform length.
    
    Args:
        tokens: List of token lists
        vectors: List of vector sequences (list of numpy arrays)
        labels: List or array of labels
        max_len: Maximum sequence length
        vector_dim: Dimension of each vector
        
    Returns:
        Tuple of (X tensor, y tensor, token lists)
    """
    padded_vectors = []
    for vec_seq in vectors:
        if vec_seq is not None and len(vec_seq) > 0:
            vec_seq = np.array(vec_seq, dtype=np.float32)
            if len(vec_seq) < max_len:
                pad_len = max_len - len(vec_seq)
                vec_seq = np.pad(vec_seq, ((0, pad_len), (0, 0)), mode='constant')
            else:
                vec_seq = vec_seq[:max_len]
            padded_vectors.append(vec_seq)
        else:
            vec_seq = np.zeros((max_len, vector_dim), dtype=np.float32)
            padded_vectors.append(vec_seq)
    
    padded_vectors = np.array(padded_vectors, dtype=np.float32)
    X = torch.tensor(padded_vectors)
    y = torch.tensor(labels)
    
    return X, y, tokens


def create_model(embedding_dim: int, num_classes: int) -> Tuple[nn.Conv1d, nn.Linear]:
    """Create a 1D CNN classification model.
    
    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of output classes
        
    Returns:
        Tuple of (conv_layer, fc_layer)
    """
    conv = nn.Conv1d(
        in_channels=embedding_dim,
        out_channels=100,
        kernel_size=3,
        padding=1
    )
    fc = nn.Linear(100, num_classes)
    return conv, fc


def train_model(X: torch.Tensor, y: torch.Tensor, conv: nn.Conv1d, 
                fc: nn.Linear, num_classes: int, epochs: int = 5, 
                batch_size: int = 16, lr: float = 1e-3, 
                device: str = None):
    """Train the CNN model.
    
    Args:
        X: Input tensor (batch, seq_len, embedding_dim)
        y: Label tensor
        conv: Conv1d layer
        fc: Linear layer
        num_classes: Number of classes
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu' (auto-detected if None)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    conv = conv.to(device)
    fc = fc.to(device)
    X = X.to(device)
    y = y.to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(list(conv.parameters()) + list(fc.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            xb = xb.permute(0, 2, 1)  # (B, E, T)
            optimizer.zero_grad()
            x_conv = F.relu(conv(xb))
            x_pooled = F.adaptive_max_pool1d(x_conv, 1).squeeze(2)
            logits = fc(x_pooled)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


def get_feature_importance(x_input: torch.Tensor, conv: nn.Conv1d, 
                           fc: nn.Linear, max_tokens: int = 100,
                           tokens: list = None) -> Tuple[int, list]:
    """Get feature importance via gradient-based attribution.
    
    Args:
        x_input: Single input tensor (seq_len, embedding_dim)
        conv: Conv1d layer
        fc: Linear layer
        max_tokens: Maximum number of top tokens to return
        tokens: Optional token list for labeling
        
    Returns:
        Tuple of (predicted_class, list of (token, importance) tuples)
    """
    conv.eval()
    fc.eval()

    # Ensure input is on same device as model
    device = next(conv.parameters()).device
    x_input = x_input.to(device).unsqueeze(0).requires_grad_(True)
    xb = x_input.permute(0, 2, 1)
    
    x_conv = F.relu(conv(xb))
    x_pooled = F.adaptive_max_pool1d(x_conv, 1).squeeze(2)
    logits = fc(x_pooled)
    pred_class = logits.argmax(dim=1)
    
    logits[0, pred_class].backward()
    grads = x_input.grad.data.abs().squeeze(0)
    token_importance = grads.mean(dim=1)
    
    if tokens:
        token_scores = list(zip(tokens, token_importance.detach().cpu().numpy()))
        token_scores.sort(key=lambda x: -x[1])
        return pred_class.item(), token_scores[:max_tokens]
    
    return pred_class.item(), token_importance


def process_phrases_ngrams(df: pd.DataFrame, tokens_col: str, 
                           vectors_col: str, labels_col: str,
                           phrase_type: str = 'phrase',
                           device: str = None) -> pd.DataFrame:
    """Full pipeline: prepare data, train CNN, extract top predictive features.
    
    Args:
        df: DataFrame with tokens, vectors, and labels columns
        tokens_col: Column name containing token lists
        vectors_col: Column name containing vector lists
        labels_col: Column name containing classification labels
        phrase_type: 'phrase' or 'ngram' (for logging)
        device: 'cuda' or 'cpu'
        
    Returns:
        DataFrame with added 'top_predicted_{phrase_type}' column
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokens = df[tokens_col].tolist()
    vectors = df[vectors_col].tolist()
    labels = LabelEncoder().fit_transform(df[labels_col])
    
    # Prepare data
    X, y, all_tokens = prepare_data(tokens, vectors, labels)
    embedding_dim = X.shape[2]
    num_classes = len(set(y.tolist()))
    
    # Create and train model
    conv, fc = create_model(embedding_dim, num_classes)
    conv, fc = conv.to(device), fc.to(device)
    train_model(X, y, conv, fc, num_classes)
    
    # Get top predictive features for each row
    top_predicted = []
    for i in range(len(df)):
        pred_class, token_importance = get_feature_importance(
            X[i], conv, fc, tokens=all_tokens[i] if all_tokens else None
        )
        top_predicted.append(token_importance)
    
    col_name = f"top_predicted_{phrase_type}"
    df[col_name] = top_predicted
    
    return df
