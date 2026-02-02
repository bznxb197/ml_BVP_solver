import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from scipy.integrate import solve_bvp
from generate_dataset_V4 import ode_system, bc, build_bspline_basis, get_params

# 1. Определение архитектуры (в точности как при обучении Turbo)
class BVPNetTurbo(nn.Module):
    def __init__(self, input_dim=25, output_dim=16):
        super(BVPNetTurbo, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
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

# 2. Загрузка модели и вспомогательных объектов
print("Загрузка Turbo-модели...")
model = BVPNetTurbo()
model.load_state_dict(torch.load("model_nn_v5_turbo.pth"))
model.eval()

scalers = joblib.load("scalers_v5.pkl")
scaler_x = scalers['scaler_x']
scaler_y = scalers['scaler_y']

def run_final_test(n_cases=100):
    # Сетка для начального приближения
    x_nodes = np.linspace(0, 1, 120) 
    Phi = build_bspline_basis(x_nodes)
    
    wins = 0      # Когда NN сошлась, а Zero — нет
    total_nn_success = 0
    results = []

    print(f"Запуск финального теста на {n_cases} жестких задачах (tol=1e-5)...")
    print("-" * 70)

    for i in range(n_cases):
        p = get_params()
        # Генерируем жесткие параметры
        p[0] = 10**np.random.uniform(np.log10(5e-5), np.log10(8e-4))
        
        p_ml = p.copy()
        p_ml[0] = np.log10(p[0])
        p_s = scaler_x.transform(p_ml.reshape(1, -1))
        
        with torch.no_grad():
            y_pred_s = model(torch.FloatTensor(p_s)).numpy()
            y_coeffs = scaler_y.inverse_transform(y_pred_s)[0]
        
        y_nn = Phi @ y_coeffs
        guess_nn = np.vstack([y_nn, np.gradient(y_nn, x_nodes)])

        y_zero = np.linspace(p[1], p[2], len(x_nodes))
        guess_zero = np.vstack([y_zero, np.zeros_like(y_zero)])

        sol_nn = solve_bvp(lambda x,y: ode_system(x,y,p), lambda ya,yb: bc(ya,yb,p), 
                           x_nodes, guess_nn, tol=1e-5)
        
        sol_zero = solve_bvp(lambda x,y: ode_system(x,y,p), lambda ya,yb: bc(ya,yb,p), 
                             x_nodes, guess_zero, tol=1e-5)

        if sol_nn.success:
            total_nn_success += 1
            n_nn = sol_nn.niter
            
            if not sol_zero.success:
                wins += 1
                print(f"[WIN!] Задача {i:02d}: NN сошлась (iter={n_nn}), Zero — УПАЛ. eps={p[0]:.2e}")
                n_zero = 999
            else:
                n_zero = sol_zero.niter
            
            results.append([p[0], n_nn, n_zero])

    # Итоговый анализ
    df_res = pd.DataFrame(results, columns=['eps', 'n_nn', 'n_zero'])
    
    print("-" * 70)
    print(f"ТЕСТ ЗАВЕРШЕН.")
    print(f"Всего побед (NN спас решение): {wins}")
    print(f"Общая надежность NN старта: {(total_nn_success/n_cases)*100:.1f}%")
    
    valid = df_res[df_res['n_zero'] < 999]
    if len(valid) > 0:
        iter_save = (1 - valid['n_nn'].mean() / valid['n_zero'].mean()) * 100
        print(f"Средняя экономия итераций: {iter_save:.1f}%")
        
    return df_res

if __name__ == "__main__":
    df = run_final_test(100)
