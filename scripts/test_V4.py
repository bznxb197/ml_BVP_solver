import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from scipy.integrate import solve_bvp
from generate_dataset_V4 import ode_system, bc, build_bspline_basis, get_params

# Архитектура должна точно совпадать с ML_V5_NN.py
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.act   = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class BVPNetTurbo(nn.Module):
    def __init__(self, input_dim=25, output_dim=16, hidden=256):
        super().__init__()
        self.in_layer  = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                       nn.BatchNorm1d(hidden), nn.Dropout(0.1))
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

print("Загрузка модели...")
model = BVPNetTurbo()
model.load_state_dict(torch.load("/Users/aleksandrcernyh/numeric_ML/model_nn_v5_turbo.pth", map_location="cpu"))
model.eval()

scalers  = joblib.load("scalers_v5.pkl")
scaler_x = scalers['scaler_x']
scaler_y = scalers['scaler_y']


def run_final_test(n_cases=5000, hard_only=False, seed=42):
    """
    Параметры:
        n_cases   — количество задач
        hard_only — если True, форсируем жёсткие eps (out-of-distribution анализ)
        seed      — для воспроизводимости
    """
    # ИСПРАВЛЕНО: фиксируем seed
    np.random.seed(seed)

    x_nodes = np.linspace(0, 1, 500)  # ИСПРАВЛЕНО: единая сетка с генератором
    Phi     = build_bspline_basis(x_nodes)

    wins_nn   = 0   # NN сошлась, Zero — нет
    wins_zero = 0   # Zero сошлась, NN — нет  (ИСПРАВЛЕНО: теперь считаем явно)
    both_win  = 0
    both_fail = 0

    nn_niter_list   = []
    zero_niter_list = []
    eps_list        = []

    print(f"{'Тест: жёсткие задачи' if hard_only else 'Тест: штатное распределение'}")
    print(f"Задач: {n_cases} | tol=1e-5 | seed={seed}")
    print("-" * 70)

    for i in range(n_cases):
        p = get_params()

        if hard_only:
            # Форсируем тяжёлый диапазон epsilon
            p[0] = 10**np.random.uniform(np.log10(5e-5), np.log10(8e-4))

        # Инференс нейросети
        p_ml    = p.copy()
        p_ml[0] = np.log10(p[0])
        p_s     = scaler_x.transform(p_ml.reshape(1, -1))

        with torch.no_grad():
            y_pred_s = model(torch.FloatTensor(p_s)).numpy()
            y_coeffs = scaler_y.inverse_transform(y_pred_s)[0]

        y_nn     = Phi @ y_coeffs
        guess_nn = np.vstack([y_nn, np.gradient(y_nn, x_nodes)])

        y_zero     = np.linspace(p[1], p[2], len(x_nodes))
        guess_zero = np.vstack([y_zero, np.zeros_like(y_zero)])

        sol_nn   = solve_bvp(lambda x, y: ode_system(x, y, p),
                             lambda ya, yb: bc(ya, yb, p),
                             x_nodes, guess_nn,   tol=1e-5)
        sol_zero = solve_bvp(lambda x, y: ode_system(x, y, p),
                             lambda ya, yb: bc(ya, yb, p),
                             x_nodes, guess_zero, tol=1e-5)

        nn_ok   = sol_nn.success
        zero_ok = sol_zero.success

        eps_list.append(p[0])

        if nn_ok and zero_ok:
            both_win += 1
            nn_niter_list.append(sol_nn.niter)
            zero_niter_list.append(sol_zero.niter)
        elif nn_ok and not zero_ok:
            wins_nn += 1
            print(f"  [NN WIN #{wins_nn:03d}] задача {i:04d}: NN iter={sol_nn.niter}, Zero=FAIL | eps={p[0]:.2e}")
        elif zero_ok and not nn_ok:
            # ИСПРАВЛЕНО: явно логируем случаи поражения NN
            wins_zero += 1
        else:
            both_fail += 1

    total = n_cases
    print("-" * 70)
    print(f"РЕЗУЛЬТАТЫ ({total} задач)")
    print(f"  Оба сошлись      : {both_win:5d}  ({100*both_win/total:.1f}%)")
    print(f"  Только NN        : {wins_nn:5d}  ({100*wins_nn/total:.1f}%)  ← победы NN")
    print(f"  Только Zero      : {wins_zero:5d}  ({100*wins_zero/total:.1f}%)  ← поражения NN")
    print(f"  Оба упали        : {both_fail:5d}  ({100*both_fail/total:.1f}%)")
    print()
    print(f"  Success Rate NN  : {100*(both_win+wins_nn)/total:.1f}%")
    print(f"  Success Rate Zero: {100*(both_win+wins_zero)/total:.1f}%")
    print(f"  Соотношение побед/поражений NN: {wins_nn}/{wins_zero}"
          + (f" = {wins_nn/wins_zero:.1f}:1" if wins_zero > 0 else " (поражений нет)"))

    if nn_niter_list:
        iter_save = (1 - np.mean(nn_niter_list) / np.mean(zero_niter_list)) * 100
        print(f"\n  Среднее итераций NN  : {np.mean(nn_niter_list):.1f}")
        print(f"  Среднее итераций Zero: {np.mean(zero_niter_list):.1f}")
        print(f"  Экономия итераций    : {iter_save:.1f}%")

    return {
        'both_win'  : both_win,
        'wins_nn'   : wins_nn,
        'wins_zero' : wins_zero,
        'both_fail' : both_fail,
        'sr_nn'     : (both_win + wins_nn) / total,
        'sr_zero'   : (both_win + wins_zero) / total,
    }


if __name__ == "__main__":
    print("=== Тест 1: штатное распределение (in-distribution) ===")
    stats_normal = run_final_test(n_cases=5000, hard_only=False, seed=42)

    print("\n=== Тест 2: только жёсткие задачи (stress-test) ===")
    stats_hard = run_final_test(n_cases=5000, hard_only=True, seed=42)