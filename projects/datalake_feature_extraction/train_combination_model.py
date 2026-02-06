#!/usr/bin/env python3
"""
train_combination_model.py

Train a model on the n-gram/phrase combination features discovered
through convolution.

This model learns which combinations of predictive n-grams appearing
in specific phrase types are most indicative of MOE vs MOP.

Features used:
- Count of predictive n-grams per phrase type
- Average predictability score per phrase type
- Presence of top discriminative patterns

Usage:
    python train_combination_model.py \
        --input data_with_convolutions.parquet \
        --output-model combination_model.pt \
        --patterns-file analysis/ngram_phrase_patterns.csv \
        --tag-col Narratives_sim_top_tag
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class CombinationClassifier(nn.Module):
    """
    Neural network for classifying based on n-gram/phrase combination features.
    """

    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=2, dropout=0.3):
        super(CombinationClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        features = self.feature_layers(x)
        return self.classifier(features)

    def get_features(self, x):
        """Get learned feature representation."""
        return self.feature_layers(x)


def build_feature_matrix(df, phrase_cols, top_patterns=None):
    """
    Build feature matrix from combination columns.

    Features:
    - predictive_ngrams_in_{phrase_type}_count
    - predictive_ngrams_in_{phrase_type}_avg_score
    - (optional) presence of top discriminative patterns
    """
    feature_cols = []

    # Count and score features per phrase type
    for phrase_col in phrase_cols:
        count_col = f'predictive_ngrams_in_{phrase_col}_count'
        score_col = f'predictive_ngrams_in_{phrase_col}_avg_score'

        if count_col in df.columns:
            feature_cols.append(count_col)
        if score_col in df.columns:
            feature_cols.append(score_col)

    if not feature_cols:
        raise ValueError("No combination feature columns found")

    print(f"Using {len(feature_cols)} feature columns")

    # Build base feature matrix
    X = df[feature_cols].fillna(0).values

    # Add top pattern presence features if provided
    if top_patterns is not None and len(top_patterns) > 0:
        print(f"Adding {len(top_patterns)} pattern presence features")

        pattern_features = []
        for idx in range(len(df)):
            row_features = []
            for pattern in top_patterns:
                # Check if this pattern appears in this row
                # Look in intersection columns for this pattern
                found = 0
                for col in df.columns:
                    if '_in_' in col and not col.endswith('_count') and not col.endswith('_score'):
                        intersections = df.iloc[idx].get(col, [])
                        if intersections:
                            for inter in intersections:
                                if inter.get('ngram') == pattern['ngram'] and \
                                   inter.get('phrase') == pattern['phrase']:
                                    found = 1
                                    break
                        if found:
                            break
                row_features.append(found)
            pattern_features.append(row_features)

        pattern_features = np.array(pattern_features)
        X = np.hstack([X, pattern_features])

        feature_cols.extend([f"pattern_{i}" for i in range(len(top_patterns))])

    return X, feature_cols


def train_model(X_train, y_train, X_val, y_val, num_classes, epochs=100, device='cpu'):
    """Train the combination classifier."""

    input_dim = X_train.shape[1]

    model = CombinationClassifier(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        num_classes=num_classes,
        dropout=0.3
    ).to(device)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += len(yb)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_correct = (val_logits.argmax(dim=1) == y_val_t).sum().item()
            val_acc = val_correct / len(y_val_t)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Acc={train_correct/train_total:.4f}, "
                  f"Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


def analyze_feature_importance(model, feature_names, X, y, device='cpu'):
    """Analyze which features are most important for classification."""

    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    X_t.requires_grad_(True)

    # Forward pass
    logits = model(X_t)
    pred_classes = logits.argmax(dim=1)

    # Compute gradients for predicted classes
    importances = []

    for i in range(len(X)):
        model.zero_grad()
        logits[i, pred_classes[i]].backward(retain_graph=True)

        grad = X_t.grad[i].abs().cpu().numpy()
        importances.append(grad)

        X_t.grad.zero_()

    # Average importance per feature
    avg_importance = np.mean(importances, axis=0)

    # Sort by importance
    feature_importance = list(zip(feature_names, avg_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance


def main():
    parser = argparse.ArgumentParser(
        description='Train model on n-gram/phrase combination features'
    )
    parser.add_argument('--input', '-i', required=True, help='Input parquet with convolution features')
    parser.add_argument('--output-model', '-o', required=True, help='Output model path')
    parser.add_argument('--patterns-file', default=None, help='CSV file with top patterns')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag', help='Tag column')
    parser.add_argument('--top-n-patterns', type=int, default=50,
                       help='Number of top patterns to use as features')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")

    # Filter to labeled rows
    mask = df[args.tag_col].notna() & df[args.tag_col].isin(['MOE', 'MOP'])
    df_labeled = df[mask].copy()
    print(f"Labeled rows: {len(df_labeled)}")

    # Load top patterns if provided
    top_patterns = None
    if args.patterns_file and Path(args.patterns_file).exists():
        print(f"Loading patterns from: {args.patterns_file}")
        patterns_df = pd.read_csv(args.patterns_file)
        patterns_df = patterns_df.sort_values('discriminative_score', ascending=False)
        top_patterns = patterns_df.head(args.top_n_patterns).to_dict('records')
        print(f"Using top {len(top_patterns)} discriminative patterns")

    # Identify phrase columns
    phrase_cols = [
        'noun_phrases', 'verb_phrases', 'adjective_phrases', 'adverb_phrases',
        'prepositional_phrases', 'gerund_phrases', 'infinitive_phrases',
        'participle_phrases', 'appositive_phrases'
    ]
    phrase_cols = [p for p in phrase_cols if p in df.columns]

    # Build feature matrix
    print("\nBuilding feature matrix...")
    X, feature_names = build_feature_matrix(df_labeled, phrase_cols, top_patterns)
    print(f"Feature matrix shape: {X.shape}")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df_labeled[args.tag_col].values)
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {np.bincount(y)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, stratify=y, random_state=42
    )

    print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

    # Train model
    print("\nTraining combination model...")
    model, best_val_acc = train_model(
        X_train, y_train, X_test, y_test,
        num_classes=len(le.classes_),
        epochs=args.epochs,
        device=device
    )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Final evaluation
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=le.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_preds)
    print(f"             Predicted")
    print(f"             {'MOE':>6} {'MOP':>6}")
    print(f"Actual MOE  {cm[0,0]:>6} {cm[0,1]:>6}")
    print(f"       MOP  {cm[1,0]:>6} {cm[1,1]:>6}")

    # Feature importance analysis
    print("\nTop 20 Most Important Features:")
    importance = analyze_feature_importance(model, feature_names, X_test, y_test, device)

    for feat, imp in importance[:20]:
        print(f"  {imp:.4f}: {feat}")

    # Save model and metadata
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names,
        'label_encoder_classes': le.classes_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'best_accuracy': best_val_acc,
        'feature_importance': importance[:50]
    }, output_path)

    print(f"\nModel saved to: {output_path}")

    # Save feature importance report
    importance_df = pd.DataFrame(importance, columns=['feature', 'importance'])
    importance_path = output_path.parent / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")


if __name__ == '__main__':
    main()
