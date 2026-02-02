import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import BSpline

# --- Конфигурация страницы ---
st.set_page_config(page_title="DeepBVP Expert System", layout="wide")

# --- Базис (из генератора) ---
DEGREE = 3
N_BASIS = 16 
knots_internal = np.linspace(0, 1, N_BASIS - DEGREE + 1)
KNOTS = np.concatenate((np.zeros(DEGREE), knots_internal, np.ones(DEGREE)))

# --- Модель (Исправленная архитектура под твои веса) ---
class BVPNetTurbo(nn.Module):
    def __init__(self, input_dim=25, output_dim=16):
        super(BVPNetTurbo, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),      # 0
            nn.GELU(),                      # 1
            nn.BatchNorm1d(512),            # 2
            nn.Dropout(0.1),                # 3
            nn.Linear(512, 256),            # 4
            nn.GELU(),                      # 5
            nn.Linear(256, 256),            # 6
            nn.GELU(),                      # 7
            nn.Linear(256, 128),            # 8
            nn.GELU(),                      # 9
            nn.Linear(128, output_dim)      # 10
        )
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_assets():
    model = BVPNetTurbo()
    model.load_state_dict(torch.load("model_nn_v5_turbo.pth", map_location="cpu"))
    model.eval()
    scalers = joblib.load("scalers_v5.pkl")
    return model, scalers

# --- Математическое ядро ---
def ode_system_logic(x, y, p):
    # Распаковка всех 25 параметров
    (eps, alpha, beta, p0, p1, p2, w1, w2, v1, v2, 
     q0, q1, q2, e1, e2, u1, u2, j, k, 
     A_src, mu, sigma, c0, c1, c2) = p
    
    px = p0 + p1*x + p2*x**2 + w1*np.sin(w2*x) + v1*np.cos(v2*x)
    qx = q0 + q1*x + q2*x**2 + e1*np.sin(e2*x) + u1*np.cos(u2*x)
    fx = A_src*np.exp(-((x - mu)/sigma)**2) + c0 + c1*x + c2*x**2
    
    return np.vstack([y[1], (fx - px*y[1] - qx*y[0] - j*y[0]**2 - k*y[0]**3) / eps])

def bc_logic(ya, yb, p):
    return np.array([ya[0] - p[1], yb[0] - p[2]])

# --- Интерфейс ---
st.title("DeepBVP: Аналитический Гибридный Решатель")

# --- Блок математики ---
st.latex(r"\varepsilon y'' + p(x)y' + q(x)y + j y^2 + k y^3 = f(x)")
with st.expander("Показать расшифровку функций (25 параметров)"):
    st.markdown("Приложение использует следующие зависимости для построения уравнения:")
    st.latex(r"p(x) = p_0 + p_1 x + p_2 x^2 + w_1 \sin(w_2 x) + v_1 \cos(v_2 x)")
    st.latex(r"q(x) = q_0 + q_1 x + q_2 x^2 + e_1 \sin(e_2 x) + u_1 \cos(u_2 x)")
    st.latex(r"f(x) = A_{src} e^{-\frac{(x-\mu)^2}{\sigma^2}} + c_0 + c_1 x + c_2 x^2")

# --- Сайдбар ---
st.sidebar.header("Управление")

if st.sidebar.button("Случайная задача"):
    st.session_state.p_rand = {
        "eps_log": np.random.uniform(-4, -1.3),
        "alpha": np.random.uniform(-2, 2), "beta": np.random.uniform(-2, 2),
        "p_coeffs": np.random.uniform(-3, 3, 3), 
        "w_params": [np.random.uniform(-1.5, 1.5), 10**np.random.uniform(0, 0.7)],
        "v_params": [np.random.uniform(-1.5, 1.5), 10**np.random.uniform(0, 0.7)],
        "q_coeffs": np.random.uniform(-3, 3, 3),
        "e_params": [np.random.uniform(-1.5, 1.5), 10**np.random.uniform(0, 0.7)],
        "u_params": [np.random.uniform(-1.5, 1.5), 10**np.random.uniform(0, 0.7)],
        "nonlin": [np.random.uniform(-2.0, 2.0), np.random.uniform(-1.0, 1.0)],
        "src_main": [np.random.uniform(-4, 4), np.random.uniform(0.2, 0.8), 10**np.random.uniform(-2.0, -0.5)],
        "src_poly": np.random.uniform(-3, 3, 3)
    }

pr = st.session_state.get('p_rand', {})
p_final = []

# Формирование списка параметров из UI
with st.sidebar.expander("Базовые (ε, ГУ)", expanded=True):
    p_final.append(st.slider("log10(ε)", -4.0, -1.3, pr.get("eps_log", -2.0)))
    p_final.append(st.slider("α (y(0))", -2.0, 2.0, pr.get("alpha", 0.0)))
    p_final.append(st.slider("β (y(1))", -2.0, 2.0, pr.get("beta", 1.0)))

with st.sidebar.expander("Коэффициенты p(x)"):
    p_final.extend([st.number_input(f"p{i}", -3.0, 3.0, float(pr.get("p_coeffs", [0,0,0])[i])) for i in range(3)])
    p_final.append(st.number_input("w1 (amp)", -1.5, 1.5, float(pr.get("w_params", [0,1])[0])))
    p_final.append(st.number_input("w2 (freq)", 1.0, 5.0, float(pr.get("w_params", [0,1])[1])))
    p_final.append(st.number_input("v1 (amp)", -1.5, 1.5, float(pr.get("v_params", [0,1])[0])))
    p_final.append(st.number_input("v2 (freq)", 1.0, 5.0, float(pr.get("v_params", [0,1])[1])))

with st.sidebar.expander("Коэффициенты q(x)"):
    p_final.extend([st.number_input(f"q{i}", -3.0, 3.0, float(pr.get("q_coeffs", [0,0,0])[i])) for i in range(3)])
    p_final.append(st.number_input("e1 (amp)", -1.5, 1.5, float(pr.get("e_params", [0,1])[0])))
    p_final.append(st.number_input("e2 (freq)", 1.0, 5.0, float(pr.get("e_params", [0,1])[1])))
    p_final.append(st.number_input("u1 (amp)", -1.5, 1.5, float(pr.get("u_params", [0,1])[0])))
    p_final.append(st.number_input("u2 (freq)", 1.0, 5.0, float(pr.get("u_params", [0,1])[1])))

with st.sidebar.expander("Нелинейность и Источник"):
    p_final.append(st.number_input("j (y²)", -2.0, 2.0, float(pr.get("nonlin", [0,0])[0])))
    p_final.append(st.number_input("k (y³)", -1.0, 1.0, float(pr.get("nonlin", [0,0])[1])))
    p_final.append(st.number_input("A (Gaus)", -4.0, 4.0, float(pr.get("src_main", [0,0.5,0.1])[0])))
    p_final.append(st.number_input("μ (Mean)", 0.2, 0.8, float(pr.get("src_main", [0,0.5,0.1])[1])))
    p_final.append(st.number_input("σ (Sigma)", 0.01, 0.5, float(pr.get("src_main", [0,0.5,0.1])[2])))
    p_final.extend([st.number_input(f"c{i}", -3.0, 3.0, float(pr.get("src_poly", [0,0,0])[i])) for i in range(3)])

# --- Расчет ---
if st.button("Решить задачу"):
    model, scalers = load_assets()
    x_plot = np.linspace(0, 1, 150)
    
    # Подготовка параметров для решателя
    p_num = np.array(p_final).copy()
    p_num[0] = 10**p_num[0] # eps
    
    # Предсказание нейросети
    p_s = scalers['scaler_x'].transform(np.array(p_final).reshape(1, -1))
    with torch.no_grad():
        y_c_s = model(torch.FloatTensor(p_s)).numpy()
        y_c = scalers['scaler_y'].inverse_transform(y_c_s)[0]
    
    y_ml_init = BSpline(KNOTS, y_c, DEGREE, extrapolate=False)(x_plot)
    
    # Сравнение методов
    t0 = time.time()
    res_std = solve_bvp(lambda x,y: ode_system_logic(x,y,p_num), 
                        lambda ya,yb: bc_logic(ya,yb,p_num), 
                        x_plot, np.vstack([np.linspace(p_num[1], p_num[2], 150), np.zeros(150)]), tol=1e-3)
    t_std = time.time() - t0
    
    t1 = time.time()
    guess_ml = np.vstack([y_ml_init, np.gradient(y_ml_init, x_plot)])
    res_ml = solve_bvp(lambda x,y: ode_system_logic(x,y,p_num), 
                       lambda ya,yb: bc_logic(ya,yb,p_num), 
                       x_plot, guess_ml, tol=1e-3)
    t_ml = time.time() - t1

    # Визуализация
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("График решения y(x)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_plot, y_ml_init, 'r--', label='ML Start Guess', alpha=0.5)
        if res_ml.success:
            ax.plot(res_ml.x, res_ml.y[0], 'b-', linewidth=2, label='Final Solution')
        ax.legend(); ax.grid(True, alpha=0.2); st.pyplot(fig)
    
    with c2:
        st.subheader("Эффективность")
        st.table({
            "Метод": ["Стандарт", "DeepBVP"],
            "Итерации": [res_std.niter if res_std.success else "Fail", res_ml.niter],
            "Время (сек)": [f"{t_std:.4f}", f"{t_ml:.4f}"]
        })
    
    st.subheader("Аналитическое решение (сплайн-коэффициенты)")
    st.json({f"c_{i+1}": float(val) for i, val in enumerate(y_c)})
