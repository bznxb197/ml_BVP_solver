import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# --- 1. Определение архитектуры (V5 Turbo) ---
class BVPNetTurbo(nn.Module):
    def __init__(self, input_dim=25, output_dim=16):
        super(BVPNetTurbo, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(),
            nn.BatchNorm1d(512), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

# --- 2. Функции базиса ---
def build_bspline_basis(x, n_bases=16):
    from scipy.interpolate import BSpline
    knots = np.linspace(0, 1, n_bases - 2)
    knots = np.pad(knots, (3, 3), mode='edge')
    basis_funcs = []
    for i in range(n_bases):
        coeffs = np.zeros(n_bases)
        coeffs[i] = 1
        spline = BSpline(knots, coeffs, k=3)
        basis_funcs.append(spline(x))
    return np.array(basis_funcs).T

# --- 3. Загрузка ресурсов ---
@st.cache_resource
def load_assets():
    model = BVPNetTurbo()
    # Загружаем на CPU для сервера
    model.load_state_dict(torch.load("model_nn_v5_turbo.pth", map_location=torch.device('cpu')))
    model.eval()
    scalers = joblib.load("scalers_v5.pkl")
    return model, scalers

# --- 4. Настройка страницы ---
st.set_page_config(page_title="DeepBVP Solver", page_icon="📈", layout="centered")

st.title("🚀 Smart Boundary Value Problem Solver")
st.markdown("""
### Нейросетевое ускорение численных методов
Этот инструмент решает **жесткие краевые задачи** (на примере уравнения Блазиуса). 
Нейросеть генерирует начальное приближение, которое позволяет классическому решателю сойтись мгновенно.
""")

# Сайдбар
st.sidebar.header("Настройки задачи")
eps = st.sidebar.select_slider("Параметр ε (вязкость)", options=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005], value=0.005)
y_left = st.sidebar.number_input("y(0) [Wall speed]", value=0.0, step=0.1)
y_right = st.sidebar.number_input("y(1) [Free stream]", value=1.0, step=0.1)

if st.sidebar.button("Рассчитать решение"):
    with st.spinner('Нейросеть генерирует старт...'):
        model, scalers = load_assets()
        x_nodes = np.linspace(0, 1, 100)
        Phi = build_bspline_basis(x_nodes)
        
        # Инференс
        full_params = np.zeros(25)
        full_params[0], full_params[1], full_params[2] = np.log10(eps), y_left, y_right
        p_s = scalers['scaler_x'].transform(full_params.reshape(1, -1))
        
        with torch.no_grad():
            y_coeffs_s = model(torch.FloatTensor(p_s)).numpy()
            y_coeffs = scalers['scaler_y'].inverse_transform(y_coeffs_s)[0]
        
        y_nn = Phi @ y_coeffs
        guess_nn = np.vstack([y_nn, np.gradient(y_nn, x_nodes)])
        
        # Численное уточнение
        def ode(x, y): return np.vstack([y[1], -y[0]*y[1]/eps])
        def bc(ya, yb): return np.array([ya[0]-y_left, yb[0]-y_right])
        
        res = solve_bvp(ode, bc, x_nodes, guess_nn, tol=1e-5)

    # --- Результаты ---
    if res.success:
        st.success(f"Решение найдено! Итераций решателя: **{res.niter}**")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_nodes, y_nn, 'r--', alpha=0.5, label='ML Guess (Start)')
        ax.plot(res.x, res.y[0], 'b-', linewidth=2, label='Final Numeric Solution')
        ax.set_title(f"Профиль скорости (ε = {eps})")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
        
        st.pyplot(fig)
        
        st.info(f"💡 Без нейросети при таком ε обычный метод мог бы потратить 20+ итераций или не сойтись вовсе.")
    else:
        st.error("К сожалению, решатель не сошелся. Попробуйте увеличить ε.")