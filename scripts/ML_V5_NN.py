import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Воспроизводимость
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Загрузка, фильтрация выбросов, подготовка
df = pd.read_csv("dataset_V4.csv")
Y_raw = df.iloc[:, 25:].values
mask  = (np.abs(Y_raw) < 10).all(axis=1)
df    = df[mask]
print(f"После фильтрации выбросов: {len(df)} примеров (удалено {(~mask).sum()})")

X = df.iloc[:, :25].values
Y = df.iloc[:, 25:].values
print(f"Датасет: {len(df)} примеров, {X.shape[1]} входных признаков, {Y.shape[1]} выходов")

# StandardScaler для X, RobustScaler для Y (устойчив к оставшимся выбросам)
scaler_x = StandardScaler()
scaler_y = RobustScaler()

X_s = scaler_x.fit_transform(X)
Y_s = scaler_y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_s, Y_s, test_size=0.15, random_state=SEED
)
_, _, Y_train_real, Y_test_real = train_test_split(
    X, Y, test_size=0.15, random_state=SEED
)

X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train)
X_test_t  = torch.FloatTensor(X_test)
Y_test_t  = torch.FloatTensor(Y_test)

# 2. Архитектура: residual blocks без Dropout
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class BVPNetTurbo(nn.Module):
    def __init__(self, input_dim=25, output_dim=16, hidden=256):
        super().__init__()
        # Dropout убран: на физических задачах с чистыми данными он мешает
        self.in_layer  = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden)
        )
        self.res1      = ResidualBlock(hidden)
        self.res2      = ResidualBlock(hidden)
        self.res3      = ResidualBlock(hidden)
        self.out_layer = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.out_layer(x)

model = BVPNetTurbo()
print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# CosineAnnealingLR: плавно снижает LR без преждевременного обрыва
epochs    = 500
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# 3. Обучение с early stopping
batch_size           = 128
EARLY_STOP_PATIENCE  = 100
best_loss            = float('inf')
patience_counter     = 0

print("Запуск обучения нейросети...")
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_t.size(0))

    for i in range(0, X_train_t.size(0), batch_size):
        idx              = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_t[idx], Y_train_t[idx]
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test_t), Y_test_t).item()

    scheduler.step()  # cosine не принимает аргумент

    if test_loss < best_loss:
        best_loss        = test_loss
        patience_counter = 0
        torch.save(model.state_dict(), "model_nn_v5_turbo.pth")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stop на эпохе {epoch+1}. Лучший loss: {best_loss:.6f}")
            break

    if (epoch + 1) % 50 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:4d} | Test Loss: {test_loss:.6f} | LR: {lr:.2e} | Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

# 4. Финальные метрики в реальном пространстве
model.load_state_dict(torch.load("model_nn_v5_turbo.pth"))
model.eval()

with torch.no_grad():
    y_pred_s = model(X_test_t).numpy()

y_pred_real = scaler_y.inverse_transform(y_pred_s)

mse_real = mean_squared_error(Y_test_real, y_pred_real)
r2_real  = r2_score(Y_test_real, y_pred_real)

print(f"\n=== Финальные метрики (реальное пространство) ===")
print(f"MSE : {mse_real:.2e}")
print(f"RMSE: {np.sqrt(mse_real):.2e}")
print(f"R²  : {r2_real:.4f}")

print(f"\nR² по каждому из 16 коэффициентов сплайна:")
for i in range(16):
    r2i = r2_score(Y_test_real[:, i], y_pred_real[:, i])
    print(f"  basis[{i:02d}]: R² = {r2i:.4f}")

# 5. Сохранение артефактов
joblib.dump({'scaler_x': scaler_x, 'scaler_y': scaler_y}, "scalers_v5.pkl")
print("\nМодель и скейлеры сохранены.")