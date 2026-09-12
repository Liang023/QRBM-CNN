# ============================================================
# Improved QRBM-CNN Financial Fraud Detection
# Dataset: ULB Credit Card Fraud Detection
# ============================================================

import os
import json
import random
import copy
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve
)


from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

warnings.filterwarnings("ignore")


SEED = 42

DATA_PATH = r"creditcard.csv"

RESULT_DIR = "./fraud_results_ulb_fixed"

os.makedirs(RESULT_DIR, exist_ok=True)


BATCH_SIZE = 1024
CNN_EPOCHS = 60
CNN_LR = 0.001
WEIGHT_DECAY = 1e-3
EARLY_STOP_PATIENCE = 15

RBM_HIDDEN_DIM = 24
RBM_EPOCHS = 15
RBM_LR = 0.03
RBM_REG = 1e-6
RBM_BATCH_SIZE = 256
RBM_NORMAL_SAMPLE_SIZE = 20000


BASELINE_POS_WEIGHT = 5.0
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0



def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 100)
print(f"Device: {device}")
print("=" * 100)


df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nClass distribution:")
print(df["Class"].value_counts())

fraud_ratio = df["Class"].mean()
print(f"\nFraud ratio = {fraud_ratio:.6f}")
print(f"Fraud percentage = {fraud_ratio * 100:.4f}%")



feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

X = df[feature_cols].values.astype(np.float32)
y = df["Class"].values.astype(np.int64)

print("\nX shape:", X.shape)
print("Normal:", np.sum(y == 0))
print("Fraud:", np.sum(y == 1))


indices = np.arange(len(y))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.40,
    stratify=y,
    random_state=SEED
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=y[temp_idx],
    random_state=SEED
)

X_train_raw = X[train_idx]
X_val_raw = X[val_idx]
X_test_raw = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

print("\n" + "=" * 100)
print("DATA SPLIT (60/20/20)")
print("=" * 100)
print("Train:", len(y_train), "Fraud:", np.sum(y_train == 1))
print("Validation:", len(y_val), "Fraud:", np.sum(y_val == 1))
print("Test:", len(y_test), "Fraud:", np.sum(y_test == 1))


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val_std = scaler.transform(X_val_raw).astype(np.float32)
X_test_std = scaler.transform(X_test_raw).astype(np.float32)


rbm_scaler = MinMaxScaler(feature_range=(0, 1))
X_train_rbm = rbm_scaler.fit_transform(X_train_std).astype(np.float32)
X_val_rbm = rbm_scaler.transform(X_val_std).astype(np.float32)
X_test_rbm = rbm_scaler.transform(X_test_std).astype(np.float32)


fraud_train_indices = np.where(y_train == 1)[0]
normal_train_indices = np.where(y_train == 0)[0]

rng = np.random.default_rng(SEED)

normal_sample_size = min(RBM_NORMAL_SAMPLE_SIZE, len(normal_train_indices))
sampled_normal_indices = rng.choice(
    normal_train_indices,
    size=normal_sample_size,
    replace=False
)

rbm_pretrain_indices = np.concatenate([fraud_train_indices, sampled_normal_indices])
rng.shuffle(rbm_pretrain_indices)

X_rbm_pretrain = X_train_rbm[rbm_pretrain_indices]

print("\n" + "=" * 100)
print("QRBM PRETRAINING SET")
print("=" * 100)
print("Total:", len(X_rbm_pretrain))
print("Fraud:", len(fraud_train_indices))
print("Normal:", normal_sample_size)



num_visible = X_train_rbm.shape[1]
num_hidden = RBM_HIDDEN_DIM

print("\n" + "=" * 100)
print("QRBM TRAINING")
print("=" * 100)
print(f"Visible = {num_visible}")
print(f"Hidden  = {num_hidden}")

rbm = RestrictedBoltzmannMachine(
    num_visible=num_visible,
    num_hidden=num_hidden
).to(device)

sampler = SimulatedAnnealingOptimizer(size_limit=128)

rbm_optimizer = SGD(
    rbm.parameters(),
    lr=RBM_LR,
    momentum=0.9
)

rbm_tensor = torch.tensor(X_rbm_pretrain, dtype=torch.float32)
rbm_dataset = TensorDataset(rbm_tensor)
rbm_loader = DataLoader(
    rbm_dataset,
    batch_size=RBM_BATCH_SIZE,
    shuffle=True
)


rbm_history = []
rbm.train()

# Initialize persistent chains
persistent_chains = None

for epoch in range(RBM_EPOCHS):
    total_loss = 0.0

    for batch_x, in rbm_loader:
        batch_x = batch_x.to(device)

        try:
            hidden_states = rbm.get_hidden(batch_x, requires_grad=True)
        except TypeError:
            hidden_states = rbm.get_hidden(batch_x)

        # Use persistent chains
        if persistent_chains is None:
            persistent_chains = torch.randn_like(hidden_states)

        sampled_states = rbm.sample(sampler)
        # Update persistent chains
        persistent_chains = sampled_states.detach()

        rbm_optimizer.zero_grad()

        objective = rbm.objective(hidden_states, sampled_states)

        regularization = RBM_REG * torch.sum(rbm.quadratic_coef ** 2)
        loss = objective + regularization

        loss.backward()
        torch.nn.utils.clip_grad_norm_(rbm.parameters(), max_norm=5.0)
        rbm_optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(rbm_loader), 1)
    rbm_history.append(avg_loss)

    print(f"QRBM Epoch {epoch + 1:03d}/{RBM_EPOCHS} | Objective = {avg_loss:.6f}")



def extract_rbm_features(rbm_model, X_data, batch_size=4096):
    rbm_model.eval()
    all_features = []

    for start in range(0, len(X_data), batch_size):
        end = min(start + batch_size, len(X_data))
        batch = torch.tensor(X_data[start:end], dtype=torch.float32).to(device)

        with torch.no_grad():
            try:
                output = rbm_model.get_hidden(batch, requires_grad=False)
            except TypeError:
                output = rbm_model.get_hidden(batch)

            # Extract both visible and hidden features
            if output.shape[1] >= num_visible + num_hidden:
                visible_features = output[:, :num_visible]
                hidden_features = output[:, num_visible:num_visible + num_hidden]
            else:
                # Fallback: just use hidden features
                hidden_features = output[:, -num_hidden:]
                visible_features = torch.zeros((batch.shape[0], num_visible))

            # Concatenate visible and hidden features
            combined = torch.cat([visible_features, hidden_features], dim=1)

        all_features.append(combined.cpu().numpy())

    return np.vstack(all_features).astype(np.float32)


print("\nExtracting QRBM features...")

H_train = extract_rbm_features(rbm, X_train_rbm)
H_val = extract_rbm_features(rbm, X_val_rbm)
H_test = extract_rbm_features(rbm, X_test_rbm)

print("H_train:", H_train.shape)
print("H_val:", H_val.shape)
print("H_test:", H_test.shape)


X_train_joint = np.hstack([X_train_std, H_train]).astype(np.float32)
X_val_joint = np.hstack([X_val_std, H_val]).astype(np.float32)
X_test_joint = np.hstack([X_test_std, H_test]).astype(np.float32)

print("\nJoint dimension:", X_train_joint.shape[1])



class FraudCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Simplified projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Simplified CNN
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )

        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        logits = self.classifier(x)
        return logits.squeeze(1)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (focal_weight * alpha_weight * ce_loss).mean()




def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_true, predictions)
    balanced_accuracy = balanced_accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, probabilities)
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, probabilities)
    except Exception:
        pr_auc = np.nan

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }



def search_best_threshold(y_true, probabilities):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)

    # Handle edge cases
    if len(thresholds) == 0:
        return 0.5, pd.DataFrame()

    # Ensure consistent lengths
    max_len = min(len(precisions) - 1, len(recalls) - 1, len(thresholds))
    if max_len <= 0:
        return 0.5, pd.DataFrame()

    precisions = precisions[:max_len]
    recalls = recalls[:max_len]
    thresholds = thresholds[:max_len]

    # Avoid division by zero
    f1_scores = 2.0 * precisions * recalls / (precisions + recalls + 1e-12)

    best_index = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_index])

    threshold_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions,
        "recall": recalls,
        "f1": f1_scores
    })

    return best_threshold, threshold_df




def build_loader(X_data, y_data, shuffle=False):
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )



def predict_model(model, loader):
    model.eval()
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probabilities = torch.sigmoid(logits)
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    return np.asarray(all_targets), np.asarray(all_probabilities)



def train_cnn_model(
        model_name,
        X_train_data,
        X_val_data,
        X_test_data,
        use_focal=False,
        adaptive_threshold=True
):
    print("\n" + "=" * 100)
    print("Training:", model_name)
    print("=" * 100)

    train_loader = build_loader(X_train_data, y_train, shuffle=True)
    val_loader = build_loader(X_val_data, y_val, shuffle=False)
    test_loader = build_loader(X_test_data, y_test, shuffle=False)

    model = FraudCNN(input_dim=X_train_data.shape[1]).to(device)

    positive_num = np.sum(y_train == 1)
    negative_num = np.sum(y_train == 0)
    raw_ratio = negative_num / positive_num

    print(f"N_negative = {negative_num}")
    print(f"N_positive = {positive_num}")
    print(f"Raw imbalance ratio = {raw_ratio:.2f}")

    # Loss
    if use_focal:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        print(f"Loss = Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    else:
        pos_weight = torch.tensor([BASELINE_POS_WEIGHT], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss = Weighted BCE, pos_weight={BASELINE_POS_WEIGHT}")

    optimizer = Adam(model.parameters(), lr=CNN_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    # Best model selection
    best_state = None
    best_val_pr_auc = -1.0
    best_val_auc = -1.0
    best_epoch = 0
    early_counter = 0
    history = []

    # Training loop
    for epoch in range(CNN_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)

        # Validation with adaptive threshold
        _, val_prob = predict_model(model, val_loader)

        if adaptive_threshold:
            val_threshold, _ = search_best_threshold(y_val, val_prob)
        else:
            val_threshold = 0.5

        val_result = calculate_metrics(y_val, val_prob, threshold=val_threshold)

        val_pr_auc = val_result["pr_auc"]
        val_auc = val_result["roc_auc"]

        scheduler.step(val_pr_auc)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_threshold": val_threshold,
            "val_accuracy": val_result["accuracy"],
            "val_balanced_accuracy": val_result["balanced_accuracy"],
            "val_precision": val_result["precision"],
            "val_recall": val_result["recall"],
            "val_f1": val_result["f1"],
            "val_roc_auc": val_auc,
            "val_pr_auc": val_pr_auc
        })

        # Best epoch selection
        is_better = False
        if val_pr_auc > best_val_pr_auc:
            is_better = True
        elif np.isclose(val_pr_auc, best_val_pr_auc) and val_auc > best_val_auc:
            is_better = True

        if is_better:
            best_val_pr_auc = val_pr_auc
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            early_counter = 0
        else:
            early_counter += 1

        print(
            f"Epoch {epoch + 1:03d}/{CNN_EPOCHS}"
            f" | Loss={train_loss:.6f}"
            f" | F1={val_result['f1']:.4f}"
            f" | PR-AUC={val_pr_auc:.4f}"
            f" | Threshold={val_threshold:.4f}"
        )

        if early_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    print(f"\nBest Epoch: {best_epoch}")
    print(f"Best Validation PR-AUC: {best_val_pr_auc:.6f}")
    print(f"Best Validation ROC-AUC: {best_val_auc:.6f}")

    # Final validation with search
    _, val_prob = predict_model(model, val_loader)
    best_threshold, threshold_df = search_best_threshold(y_val, val_prob)

    print(f"Selected Threshold = {best_threshold:.6f}")

    # Test
    _, test_prob = predict_model(model, test_loader)
    test_result = calculate_metrics(y_test, test_prob, threshold=best_threshold)
    test_result["model"] = model_name
    test_result["best_epoch"] = best_epoch
    test_result["Focal"] = use_focal
    test_result["AdaptiveThreshold"] = adaptive_threshold

    print("\nTest Results")
    print(f"Accuracy          : {test_result['accuracy']:.6f}")
    print(f"Balanced Accuracy : {test_result['balanced_accuracy']:.6f}")
    print(f"Precision         : {test_result['precision']:.6f}")
    print(f"Recall            : {test_result['recall']:.6f}")
    print(f"F1                : {test_result['f1']:.6f}")
    print(f"ROC-AUC           : {test_result['roc_auc']:.6f}")
    print(f"PR-AUC            : {test_result['pr_auc']:.6f}")
    print("\nConfusion Matrix:")
    print(np.array([[test_result["tn"], test_result["fp"]],
                    [test_result["fn"], test_result["tp"]]]))

    # Save
    safe_name = model_name.replace(" ", "_").replace("+", "plus")
    pd.DataFrame(history).to_csv(
        os.path.join(RESULT_DIR, safe_name + "_history.csv"),
        index=False
    )
    if adaptive_threshold and len(threshold_df) > 0:
        threshold_df.to_csv(
            os.path.join(RESULT_DIR, safe_name + "_threshold.csv"),
            index=False
        )
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, safe_name + ".pth"))

    return test_result, model, val_prob, test_prob



def run_ml_baselines():
    print("\n" + "=" * 100)
    print("TRADITIONAL ML BASELINES (WEAK + STRONG)")
    print("=" * 100)

    # WEAK BASELINES - expected to perform poorly
    weak_models = {
        "ZeroR (Majority Class)": DummyClassifier(
            strategy="most_frequent",
            random_state=SEED
        ),
        "Random Guess": DummyClassifier(
            strategy="uniform",
            random_state=SEED
        ),
        "Decision Tree (No Prune)": DecisionTreeClassifier(
            max_depth=None,  # No pruning - severe overfitting
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=SEED
        ),
        "Naive Bayes": GaussianNB(),
        "Logistic (Unweighted)": LogisticRegression(
            max_iter=3000,
            class_weight=None,  # No class weighting - biased to majority
            solver="liblinear",
            random_state=SEED
        )
    }

    # STRONG BASELINES - expected to perform better
    strong_models = {
        "Logistic (Balanced)": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED
        ),
        "MLP (32,16)": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=300,
            early_stopping=True,
            batch_size=1024,
            random_state=SEED
        )
    }

    # Combine all models
    all_models = {**weak_models, **strong_models}

    results = []

    for model_name, model in all_models.items():
        print("\n" + "-" * 80)
        print("Training:", model_name)
        print("-" * 80)

        model.fit(X_train_std, y_train)

        # For models without predict_proba (e.g., DummyClassifier)
        try:
            val_prob = model.predict_proba(X_val_std)[:, 1]
        except AttributeError:
            # Some models might only have decision_function
            try:
                val_scores = model.decision_function(X_val_std)
                # Normalize to [0,1] for probability-like scores
                val_prob = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-12)
            except:
                # Fallback for DummyClassifier with strategy='most_frequent'
                val_prob = model.predict_proba(X_val_std)[:, 1]

        best_threshold, _ = search_best_threshold(y_val, val_prob)

        # Test prediction
        try:
            test_prob = model.predict_proba(X_test_std)[:, 1]
        except AttributeError:
            try:
                test_scores = model.decision_function(X_test_std)
                test_prob = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-12)
            except:
                test_prob = model.predict_proba(X_test_std)[:, 1]

        result = calculate_metrics(y_test, test_prob, threshold=best_threshold)
        result["model"] = model_name
        result["best_epoch"] = None
        result["Focal"] = False
        result["AdaptiveThreshold"] = True

        results.append(result)

        print(f"F1={result['f1']:.4f}"
              f" | Precision={result['precision']:.4f}"
              f" | Recall={result['recall']:.4f}"
              f" | Threshold={result['threshold']:.6f}")

    return results


# ============================================================
# 22. Run Experiments
# ============================================================

all_results = []

# 22.1 Traditional ML (WEAK + STRONG baselines)
ml_results = run_ml_baselines()
all_results.extend(ml_results)

# 22.2 Plain CNN
plain_cnn_result, _, _, _ = train_cnn_model(
    model_name="Plain CNN",
    X_train_data=X_train_std,
    X_val_data=X_val_std,
    X_test_data=X_test_std,
    use_focal=False,
    adaptive_threshold=True
)
all_results.append(plain_cnn_result)

# 22.3 QRBM + CNN
qrbm_cnn_result, _, _, _ = train_cnn_model(
    model_name="QRBM + CNN",
    X_train_data=X_train_joint,
    X_val_data=X_val_joint,
    X_test_data=X_test_joint,
    use_focal=False,
    adaptive_threshold=True
)
all_results.append(qrbm_cnn_result)

# 22.4 QRBM + CNN + Focal Loss (replaces PGA)
qrbm_focal_result, qrbm_focal_model, qrbm_focal_val_prob, qrbm_focal_test_prob = train_cnn_model(
    model_name="QRBM + CNN + Focal",
    X_train_data=X_train_joint,
    X_val_data=X_val_joint,
    X_test_data=X_test_joint,
    use_focal=True,
    adaptive_threshold=False
)
all_results.append(qrbm_focal_result)

# 22.5 Full Model: Reuse Focal model + Adaptive Threshold
print("\n" + "=" * 100)
print("FULL MODEL: REUSE FOCAL MODEL + ADAPTIVE THRESHOLD")
print("=" * 100)

full_threshold, full_threshold_df = search_best_threshold(y_val, qrbm_focal_val_prob)
full_result = calculate_metrics(y_test, qrbm_focal_test_prob, threshold=full_threshold)
full_result["model"] = "QRBM + CNN + Focal + Adaptive Threshold"
full_result["best_epoch"] = qrbm_focal_result["best_epoch"]
full_result["Focal"] = True
full_result["AdaptiveThreshold"] = True

all_results.append(full_result)

print(f"\nSelected Threshold = {full_threshold:.6f}")
print(f"F1 = {full_result['f1']:.6f}")
print("\nConfusion Matrix:")
print(np.array([[full_result["tn"], full_result["fp"]],
                [full_result["fn"], full_result["tp"]]]))



# Save QRBM
torch.save({
    "rbm_state_dict": rbm.state_dict(),
    "num_visible": num_visible,
    "num_hidden": num_hidden,
    "history": rbm_history
}, os.path.join(RESULT_DIR, "qrbm_model.pth"))

pd.DataFrame({
    "epoch": np.arange(1, len(rbm_history) + 1),
    "objective": rbm_history
}).to_csv(os.path.join(RESULT_DIR, "qrbm_history.csv"), index=False)

# Final comparison
result_df = pd.DataFrame(all_results)

display_columns = [
    "model", "accuracy", "balanced_accuracy",
    "precision", "recall", "f1",
    "roc_auc", "pr_auc", "threshold",
    "tn", "fp", "fn", "tp"
]

print("\n" + "=" * 150)
print("FINAL MODEL COMPARISON")
print("=" * 150)
print(result_df[display_columns].to_string(index=False))

result_df.to_csv(os.path.join(RESULT_DIR, "model_comparison.csv"), index=False)

# Configuration
experiment_config = {
    "dataset": DATA_PATH,
    "dataset_name": "ULB Credit Card Fraud Detection",
    "seed": SEED,
    "n_samples": int(len(y)),
    "n_features": int(X.shape[1]),
    "normal_samples": int(np.sum(y == 0)),
    "fraud_samples": int(np.sum(y == 1)),
    "fraud_ratio": float(np.mean(y)),
    "split": "60/20/20 stratified",
    "rbm_visible": int(num_visible),
    "rbm_hidden": RBM_HIDDEN_DIM,
    "rbm_epochs": RBM_EPOCHS,
    "rbm_lr": RBM_LR,
    "rbm_normal_sample_size": RBM_NORMAL_SAMPLE_SIZE,
    "cnn_epochs": CNN_EPOCHS,
    "cnn_lr": CNN_LR,
    "cnn_batch_size": BATCH_SIZE,
    "baseline_pos_weight": BASELINE_POS_WEIGHT,
    "focal_alpha": FOCAL_ALPHA,
    "focal_gamma": FOCAL_GAMMA,
    "best_epoch_rule": "Maximum validation PR-AUC with adaptive threshold; ROC-AUC tie-break",
    "threshold_rule": "Exact candidate search from validation precision-recall curve maximizing F1",
    "joint_feature_dimension": int(X_train_joint.shape[1])
}

with open(os.path.join(RESULT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
    json.dump(experiment_config, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 100)
print("Experiment Finished")
print("=" * 100)
print("Results Directory:", RESULT_DIR)