import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Загрузка и подготовка
df = pd.read_csv("dataset_V4.csv")
X = df.iloc[:, :25].values
Y = df.iloc[:, 25:].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_s = scaler_x.fit_transform(X)
Y_s = scaler_y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_s, Y_s, test_size=0.15, random_state=42)

X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train)
X_test_t = torch.FloatTensor(X_test)
Y_test_t = torch.FloatTensor(Y_test)

# 2. Улучшенная архитектура
class BVPNetTurbo(nn.Module):
    def __init__(self, input_dim=25, output_dim=16):
        super(BVPNetTurbo, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(), # Плавная активация для физики
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

model = BVPNetTurbo()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
# Снижаем LR, если лосс перестал падать
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# 3. Обучение
epochs = 400
batch_size = 128

print("Запуск Turbo-обучения нейросети...")
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_t.size(0))
    epoch_loss = 0
    
    for i in range(0, X_train_t.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_t[indices], Y_train_t[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test_t), Y_test_t)
        scheduler.step(test_loss) # Обновляем шедулер
    
    if (epoch + 1) % 50 == 0:
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss.item():.6f}, LR: {curr_lr}')

# 4. Сохранение
torch.save(model.state_dict(), "model_nn_v5_turbo.pth")
joblib.dump({'scaler_x': scaler_x, 'scaler_y': scaler_y}, "scalers_v5.pkl")

with torch.no_grad():
    y_pred_s = model(X_test_t).numpy()
    from sklearn.metrics import r2_score
    print(f"\nНовый R2 Score: {r2_score(Y_test_t.numpy(), y_pred_s):.4f}")
