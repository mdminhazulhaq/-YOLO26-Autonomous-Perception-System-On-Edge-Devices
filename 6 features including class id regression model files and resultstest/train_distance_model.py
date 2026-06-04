import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("distance_dataset_5cols.csv")

X = df[
    [
        "class_id",
        "x_center",
        "y_center",
        "width",
        "height"
    ]
].values

y = df["distance"].values.reshape(-1, 1)

print(f"\nTotal Samples: {len(df)}")

# ==========================================================
# NORMALIZATION
# ==========================================================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# ==========================================================
# TRAIN / VAL SPLIT
# ==========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

print(f"Training Samples: {len(X_train)}")
print(f"Validation Samples: {len(X_val)}")

# ==========================================================
# TENSORS
# ==========================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=32,
    shuffle=False
)

# ==========================================================
# MODEL
# ==========================================================

class DistanceRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1)

        )

    def forward(self, x):
        return self.network(x)

model = DistanceRegressionModel()

print("\nModel Architecture:\n")
print(model)

total_params = sum(p.numel() for p in model.parameters())

print(f"\nTotal Parameters: {total_params:,}")

# ==========================================================
# LOSS
# ==========================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=5,
    factor=0.5
)

# ==========================================================
# TRAINING
# ==========================================================

epochs = 100

best_val_loss = float("inf")

print("\nTraining Started...\n")

for epoch in range(epochs):

    model.train()

    train_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_distance_model.pth"
        )

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

# ==========================================================
# EVALUATION
# ==========================================================

model.load_state_dict(
    torch.load("best_distance_model.pth")
)

model.eval()

with torch.no_grad():

    predictions = model(X_val)

predictions = predictions.numpy()

predictions_m = scaler_y.inverse_transform(
    predictions
)

ground_truth_m = scaler_y.inverse_transform(
    y_val.numpy()
)

mae = np.mean(
    np.abs(
        predictions_m -
        ground_truth_m
    )
)

rmse = np.sqrt(
    np.mean(
        (
            predictions_m -
            ground_truth_m
        ) ** 2
    )
)

print("\n==========================")
print("FINAL RESULTS")
print("==========================")

print(f"Best Validation Loss : {best_val_loss:.6f}")
print(f"MAE  : {mae:.3f} m")
print(f"RMSE : {rmse:.3f} m")

# ==========================================================
# SAVE SCALERS
# ==========================================================

joblib.dump(
    scaler_X,
    "scaler_X.pkl"
)

joblib.dump(
    scaler_y,
    "scaler_y.pkl"
)

print("\nFiles Saved:")
print("best_distance_model.pth")
print("scaler_X.pkl")
print("scaler_y.pkl")