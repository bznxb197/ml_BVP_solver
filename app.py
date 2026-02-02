import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import BSpline

# --- Настройки страницы ---
st.set_page_config(page_title="DeepBVP Expert Solver", layout="wide")

# --- Константы базиса (из генератора) ---
DEGREE = 3
N_BASIS = 16 
knots_internal = np.linspace(0, 1, N_BASIS - DEGREE + 1)
KNOTS = np.concatenate((np.zeros(DEGREE), knots_internal, np.ones(DEGREE)))

# --- Модель V5 Turbo (Архитектура по твоим логам) ---
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

@st.cache_resource
def load_assets():
    model = BVPNetTurbo()
    model.load_state_dict(torch.load("model_nn_v5_turbo.pth", map_location="cpu"))
    model.eval()
    scalers = joblib.load("scalers_v5.pkl")
    return model, scalers

# --- Математическое ядро ---
def ode_system_logic(x, y, p):
    # p[0] - eps, p[1..2] - bc, далее коэффициенты функций
    eps = p[0]
    px = p[3] + p[4]*x + p[5]*x**2 + p[6]*np.sin(p[7]*x) + p[8]*np.cos(p[9]*x)
    qx = p[10] + p[11]*x + p[12]*x**2 + p[13]*np.sin(p[14]*x) + p[15]*np.cos(p[16]*x)
    fx = p[19]*np.exp(-((x - p[20])/p[21])**2) + p[22] + p[23]*x + p[24]*x**2
    return np.vstack([y[1], (fx - px*y[1] - qx*y[0] - p[17]*y[0]**2 - p[18]*y[0]**3) / eps])

def bc_logic(ya, yb, p):
    return np.array([ya[0] - p[1], yb[0] - p[2]])

# --- Интерфейс ---
st.title("DeepBVP: Hybrid Neural Solver (решатель краевых задач)")
st.latex(r"\varepsilon y'' + p(x)y' + q(x)y + j y^2 + k y^3 = f(x)")

# --- Сайдбар ---
st.sidebar.header("Параметры задачи")

if st.sidebar.button("Случайная задача"):
    st.session_state.p = [
        np.random.uniform(-4, -1.3), # log10 eps
        np.random.uniform(-2, 2),    # alpha
        np.random.uniform(-2, 2),    # beta
        *np.random.uniform(-3, 3, size=3), # p0,1,2
        np.random.uniform(-1.5, 1.5),# w1
        10**np.random.uniform(0, 0.7),# w2
        np.random.uniform(-1.5, 1.5),# v1
        10**np.random.uniform(0, 0.7),# v2
        *np.random.uniform(-3, 3, size=3), # q0,1,2
        np.random.uniform(-1.5, 1.5),# e1
        10**np.random.uniform(0, 0.7),# e2
        np.random.uniform(-1.5, 1.5),# u1
        10**np.random.uniform(0, 0.7),# u2
        np.random.uniform(-2.0, 2.0),# j
        np.random.uniform(-1.0, 1.0),# k
        np.random.uniform(-4, 4),    # A
        np.random.uniform(0.2, 0.8), # mu
        np.random.uniform(-2.0, -0.5),# log10 sigma
        *np.random.uniform(-3, 3, size=3) # c0,1,2
    ]

if 'p' not in st.session_state:
    st.session_state.p = [-2.0, 0.0, 1.0] + [0.0]*22

p_ml_input = [] # Список для нейросети (где eps в log10)

with st.sidebar.expander("Базовые параметры (ε, ГУ)", expanded=True):
    # Прямой ввод eps
    eps_val = st.number_input("Параметр ε", 0.0001, 0.1000, float(10**st.session_state.p[0]), format="%.4f", step=0.0001)
    p_ml_input.append(np.log10(eps_val))
    
    alpha = st.slider("α (y(0))", -2.0, 2.0, float(st.session_state.p[1]))
    p_ml_input.append(alpha)
    
    beta = st.slider("β (y(1))", -2.0, 2.0, float(st.session_state.p[2]))
    p_ml_input.append(beta)

with st.sidebar.expander("Функции p(x) и q(x)"):
    for i in range(3, 17):
        val = st.number_input(f"Параметр p[{i}]", value=float(st.session_state.p[i]), format="%.3f")
        p_ml_input.append(val)

with st.sidebar.expander("Нелинейность и Источник"):
    for i in range(17, 25):
        val = st.number_input(f"Параметр p[{i}]", value=float(st.session_state.p[i]), format="%.3f")
        p_ml_input.append(val)

# --- Расчетная часть ---
if st.button("Решить и сравнить"):
    model, scalers = load_assets()
    x_nodes = np.linspace(0, 1, 150)
    
    # Подготовка p_numeric (для ODE - eps обычный)
    p_numeric = np.array(p_ml_input).copy()
    p_numeric[0] = 10**p_numeric[0] # eps_val
    if p_numeric[21] < 0: p_numeric[21] = 10**p_numeric[21] # sigma
    
    # 1. СТАНДАРТНЫЙ СТАРТ (Замер времени)
    y_guess_lin = np.vstack([np.linspace(alpha, beta, 150), np.zeros(150)])
    t0 = time.time()
    res_std = solve_bvp(lambda x,y: ode_system_logic(x,y,p_numeric), 
                        lambda ya,yb: bc_logic(ya,yb,p_numeric), 
                        x_nodes, y_guess_lin, tol=1e-3)
    t_std = time.time() - t0

    # 2. ПОДГОТОВКА ML (Вне замера численного метода)
    p_s = scalers['scaler_x'].transform(np.array(p_ml_input).reshape(1, -1))
    with torch.no_grad():
        y_coeffs_s = model(torch.FloatTensor(p_s)).numpy()
        y_coeffs = scalers['scaler_y'].inverse_transform(y_coeffs_s)[0]
    
    spline = BSpline(KNOTS, y_coeffs, DEGREE, extrapolate=False)
    y_ml_init = spline(x_nodes)
    y_guess_ml_stack = np.vstack([y_ml_init, np.gradient(y_ml_init, x_nodes)])
    
    # 3. ГИБРИДНЫЙ СТАРТ (Замер времени уточнения)
    t1 = time.time()
    res_ml = solve_bvp(lambda x,y: ode_system_logic(x,y,p_numeric), 
                       lambda ya,yb: bc_logic(ya,yb,p_numeric), 
                       x_nodes, y_guess_ml_stack, tol=1e-3)
    t_ml = time.time() - t1

    # --- Визуализация ---
    col_graph, col_stats = st.columns([2, 1])
    
    with col_graph:
        st.subheader("Визуализация решения")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_nodes, np.linspace(alpha, beta, 150), 'gray', linestyle=':', alpha=0.4, label='Linear Start')
        ax.plot(x_nodes, y_ml_init, 'r--', alpha=0.6, label='ML Initial Guess')
        
        if res_ml.success:
            ax.plot(res_ml.x, res_ml.y[0], 'b-', linewidth=2.5, label='Hybrid Solution')
        if res_std.success:
            ax.plot(res_std.x, res_std.y[0], 'orange', linestyle='-.', alpha=0.6, label='Standard Result')
            
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    with col_stats:
        st.subheader("Метрики")
        st.table({
            "Показатель": ["Итерации", "Вычисления (nfev)", "Время метода (с)"],
            "Стандарт": [res_std.niter if res_std.success else "Провал", res_std.nfev if res_std.success else "—", f"{t_std:.4f}"],
            "Гибрид": [res_ml.niter if res_ml.success else "Провал", res_ml.nfev if res_ml.success else "—", f"{t_ml:.4f}"]
        })
        
        if res_std.success and res_ml.success:
            diff_fev = res_std.nfev - res_ml.nfev
            st.metric("Экономия вычислений", f"{res_std.nfev / res_ml.nfev:.1f}x", delta=f"{-diff_fev} вызовов")

    # --- Анатомия сплайна ---
    st.divider()
    st.subheader("Базисные функции решения")
    fig_b, ax_b = plt.subplots(figsize=(12, 4))
    x_fine = np.linspace(0, 1, 300)
    for i in range(N_BASIS):
        c = np.zeros(N_BASIS); c[i] = 1.0
        y_b = y_coeffs[i] * BSpline(KNOTS, c, DEGREE)(x_fine)
        ax_b.plot(x_fine, y_b, alpha=0.5, linestyle='--')
    ax_b.plot(x_fine, spline(x_fine), 'b-', linewidth=2, label="Итоговая кривая")
    st.pyplot(fig_b)

    st.write("Коэффициенты сплайна:")
    st.json({f"c_{i+1}": float(val) for i, val in enumerate(y_coeffs)})
