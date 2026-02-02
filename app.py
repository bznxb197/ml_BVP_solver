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
st.set_page_config(page_title="DeepBVP Solver Pro", layout="wide")

# --- Константы базиса (из твоего генератора) ---
DEGREE = 3
N_BASIS = 16 
knots_internal = np.linspace(0, 1, N_BASIS - DEGREE + 1)
KNOTS = np.concatenate((np.zeros(DEGREE), knots_internal, np.ones(DEGREE)))

# --- Модель V5 Turbo ---
class BVPNetTurbo(nn.Module):
    def __init__(self, input_dim=25, output_dim=16):
        super(BVPNetTurbo, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(),
            nn.BatchNorm1d(512), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
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
    # p[0] здесь уже должен быть реальный eps (не логарифм)
    eps = p[0]
    px = p[3] + p[4]*x + p[5]*x**2 + p[6]*np.sin(p[7]*x) + p[8]*np.cos(p[9]*x)
    qx = p[10] + p[11]*x + p[12]*x**2 + p[13]*np.sin(p[14]*x) + p[15]*np.cos(p[16]*x)
    fx = p[19]*np.exp(-((x - p[20])/p[21])**2) + p[22] + p[23]*x + p[24]*x**2
    return np.vstack([y[1], (fx - px*y[1] - qx*y[0] - p[17]*y[0]**2 - p[18]*y[0]**3) / eps])

def bc_logic(ya, yb, p):
    return np.array([ya[0] - p[1], yb[0] - p[2]])

# --- Интерфейс ---
st.title("DeepBVP: Hybrid Neural Solver (решатель краевых задач)")

# Описание уравнения
st.latex(r"\varepsilon y'' + p(x)y' + q(x)y + j y^2 + k y^3 = f(x)")
with st.expander("Посмотреть структуру функций"):
    st.latex(r"p(x) = p_0 + p_1 x + p_2 x^2 + w_1 \sin(w_2 x) + v_1 \cos(v_2 x)")
    st.latex(r"q(x) = q_0 + q_1 x + q_2 x^2 + e_1 \sin(e_2 x) + u_1 \cos(u_2 x)")
    st.latex(r"f(x) = A \exp\left(-\frac{(x-\mu)^2}{\sigma^2}\right) + c_0 + c_1 x + c_2 x^2")

# --- Сайдбар с 25 параметрами ---
st.sidebar.header("Параметры ОДУ")

if st.sidebar.button("Случайная задача"):
    # Генерируем в точности по get_params()
    st.session_state.p = [
        np.random.uniform(-4, -1.3), # log10 eps
        np.random.uniform(-2, 2),    # alpha
        np.random.uniform(-2, 2),    # beta
        *np.random.uniform(-3, 3, size=3), # p0,1,2
        np.random.uniform(-1.5, 1.5),# w1
        np.random.uniform(1, 5),     # w2
        np.random.uniform(-1.5, 1.5),# v1
        np.random.uniform(1, 5),     # v2
        *np.random.uniform(-3, 3, size=3), # q0,1,2
        np.random.uniform(-1.5, 1.5),# e1
        np.random.uniform(1, 5),     # e2
        np.random.uniform(-1.5, 1.5),# u1
        np.random.uniform(1, 5),     # u2
        np.random.uniform(-2.0, 2.0),# j
        np.random.uniform(-1.0, 1.0),# k
        np.random.uniform(-4, 4),    # A
        np.random.uniform(0.2, 0.8), # mu
        np.random.uniform(-2.0, -0.5),# log10 sigma
        *np.random.uniform(-3, 3, size=3) # c0,1,2
    ]

# Инициализация сессии
if 'p' not in st.session_state:
    st.session_state.p = [-2.0, 0.0, 1.0] + [0.0]*22

p_ui = []
# Группировка для удобства
with st.sidebar.expander("Граничные условия и ε", expanded=True):
    p_ui.append(st.slider("log10(ε)", -4.0, -1.3, float(st.session_state.p[0])))
    p_ui.append(st.slider("α (y0)", -2.0, 2.0, float(st.session_state.p[1])))
    p_ui.append(st.slider("β (y1)", -2.0, 2.0, float(st.session_state.p[2])))

with st.sidebar.expander("Функция p(x)"):
    for i in range(3, 10):
        p_ui.append(st.number_input(f"p[{i}]", value=float(st.session_state.p[i]), format="%.3f"))

with st.sidebar.expander("Функция q(x)"):
    for i in range(10, 17):
        p_ui.append(st.number_input(f"p[{i}]", value=float(st.session_state.p[i]), format="%.3f"))

with st.sidebar.expander("Нелинейность и Источник"):
    for i in range(17, 25):
        p_ui.append(st.number_input(f"p[{i}]", value=float(st.session_state.p[i]), format="%.3f"))

# --- Расчет ---
if st.button("Решить уравнение"):
    model, scalers = load_assets()
    x_nodes = np.linspace(0, 1, 150)
    
    # Подготовка параметров
    p_numeric = np.array(p_ui).copy()
    p_numeric[0] = 10**p_numeric[0]  # Из log10 в реальный eps
    if p_numeric[21] < 0: p_numeric[21] = 10**p_numeric[21] # sigma
    
    # 1. СТАНДАРТНЫЙ SOLVER (из прямой)
    y_guess_std = np.vstack([np.linspace(p_numeric[1], p_numeric[2], len(x_nodes)), np.zeros(len(x_nodes))])
    t0 = time.time()
    res_std = solve_bvp(lambda x,y: ode_system_logic(x,y,p_numeric), 
                        lambda ya,yb: bc_logic(ya,yb,p_numeric), 
                        x_nodes, y_guess_std, tol=1e-3)
    t_std = time.time() - t0

    # 2. HYBRID SOLVER (ML Start)
    t1 = time.time()
    # На вход модели подаем параметры как в датасете (eps уже в log10)
    p_ml = np.array(p_ui).reshape(1, -1)
    p_s = scalers['scaler_x'].transform(p_ml)
    
    with torch.no_grad():
        y_coeffs_s = model(torch.FloatTensor(p_s)).numpy()
        y_coeffs = scalers['scaler_y'].inverse_transform(y_coeffs_s)[0]
    
    # Сплайн-приближение
    spline = BSpline(KNOTS, y_coeffs, DEGREE, extrapolate=False)
    y_guess_ml = spline(x_nodes)
    y_guess_ml_stack = np.vstack([y_guess_ml, np.gradient(y_guess_ml, x_nodes)])
    
    res_ml = solve_bvp(lambda x,y: ode_system_logic(x,y,p_numeric), 
                       lambda ya,yb: bc_logic(ya,yb,p_numeric), 
                       x_nodes, y_guess_ml_stack, tol=1e-3)
    t_ml = time.time() - t1

    # --- Результаты ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Визуализация решения")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_nodes, y_guess_ml, 'r--', alpha=0.5, label='ML Initial Guess')
        if res_ml.success:
            ax.plot(res_ml.x, res_ml.y[0], 'b-', linewidth=2, label='Hybrid Solution')
        ax.set_title(f"Solution at ε = {p_numeric[0]:.2e}")
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    with col2:
        st.subheader("Аналитика")
        st.table({
            "Метод": ["Standard", "DeepBVP (Our)"],
            "Итерации": [res_std.niter if res_std.success else "Fail", res_ml.niter],
            "Время (сек)": [f"{t_std:.4f}", f"{t_ml:.4f}"]
        })

    st.subheader("Аналитическая форма сплайна")
    st.latex(r"y(x) \approx \sum_{i=1}^{16} c_i \cdot B_i(x)")
    st.write("Коэффициенты $c_i$:")
    st.json({f"c_{i+1}": float(c) for i, c in enumerate(y_coeffs)})
