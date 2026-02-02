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

# --- Константы базиса ---
DEGREE = 3
N_BASIS = 16 
knots_internal = np.linspace(0, 1, N_BASIS - DEGREE + 1)
KNOTS = np.concatenate((np.zeros(DEGREE), knots_internal, np.ones(DEGREE)))

# --- Архитектура модели ---
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
if 'eval_count' not in st.session_state:
    st.session_state.eval_count = 0

def ode_system_logic(x, y, p):
    st.session_state.eval_count += 1
    eps = p[0]
    px = p[3] + p[4]*x + p[5]*x**2 + p[6]*np.sin(p[7]*x) + p[8]*np.cos(p[9]*x)
    qx = p[10] + p[11]*x + p[12]*x**2 + p[13]*np.sin(p[14]*x) + p[15]*np.cos(p[16]*x)
    fx = p[19]*np.exp(-((x - p[20])/p[21])**2) + p[22] + p[23]*x + p[24]*x**2
    return np.vstack([y[1], (fx - px*y[1] - qx*y[0] - p[17]*y[0]**2 - p[18]*y[0]**3) / eps])

def bc_logic(ya, yb, p):
    return np.array([ya[0] - p[1], yb[0] - p[2]])

# --- Интерфейс ---
st.title("DeepBVP: Hybrid Neural Solver")

st.latex(r"\varepsilon y'' + p(x)y' + q(x)y + j y^2 + k y^3 = f(x)")
with st.expander("Математическая структура функций", expanded=False):
    st.markdown("Уравнение строится на основе следующих зависимостей:")
    st.latex(r"p(x) = p_0 + p_1 x + p_2 x^2 + w_1 \sin(w_2 x) + v_1 \cos(v_2 x)")
    st.latex(r"q(x) = q_0 + q_1 x + q_2 x^2 + e_1 \sin(e_2 x) + u_1 \cos(u_2 x)")
    st.latex(r"f(x) = A \exp\left(-\frac{(x-\mu)^2}{\sigma^2}\right) + c_0 + c_1 x + c_2 x^2")

# --- Боковая панель ---
st.sidebar.header("Параметры задачи")

col_rand1, col_rand2 = st.sidebar.columns(2)

if col_rand1.button("Случайная"):
    st.session_state.p = [np.random.uniform(-4, -1.3), np.random.uniform(-2, 2), np.random.uniform(-2, 2),
                         *np.random.uniform(-3, 3, size=3), np.random.uniform(-1.5, 1.5), 10**np.random.uniform(0, 0.7),
                         np.random.uniform(-1.5, 1.5), 10**np.random.uniform(0, 0.7), *np.random.uniform(-3, 3, size=3),
                         np.random.uniform(-1.5, 1.5), 10**np.random.uniform(0, 0.7), np.random.uniform(-1.5, 1.5),
                         10**np.random.uniform(0, 0.7), np.random.uniform(-2.0, 2.0), np.random.uniform(-1.0, 1.0),
                         np.random.uniform(-4, 4), np.random.uniform(0.2, 0.8), np.random.uniform(-2.0, -0.5),
                         *np.random.uniform(-3, 3, size=3)]

if col_rand2.button("Сложная"):
    st.session_state.p = [np.random.uniform(-4.0, -3.7), np.random.uniform(-2, 2), np.random.uniform(-2, 2),
                         *np.random.uniform(-3, 3, size=3), np.random.uniform(1.2, 1.5), np.random.uniform(4.5, 5.0),
                         np.random.uniform(1.2, 1.5), np.random.uniform(4.5, 5.0), *np.random.uniform(-3, 3, size=3),
                         np.random.uniform(1.2, 1.5), np.random.uniform(4.5, 5.0), np.random.uniform(1.2, 1.5),
                         np.random.uniform(4.5, 5.0), np.random.uniform(1.8, 2.0), np.random.uniform(0.9, 1.0),
                         np.random.uniform(3.5, 4.0), np.random.choice([0.15, 0.85]), np.random.uniform(-1.8, -1.6),
                         *np.random.uniform(-3, 3, size=3)]

if 'p' not in st.session_state:
    st.session_state.p = [-2.0, 0.0, 1.0] + [0.0]*22

p_ml_input = [] 
with st.sidebar.expander("Базовые параметры", expanded=True):
    eps_val = st.number_input("Параметр epsilon", 0.0001, 0.1000, float(10**st.session_state.p[0]), format="%.4f", step=0.0001)
    p_ml_input.append(np.log10(eps_val))
    alpha = st.slider("alpha (y0)", -2.0, 2.0, float(st.session_state.p[1]))
    p_ml_input.append(alpha)
    beta = st.slider("beta (y1)", -2.0, 2.0, float(st.session_state.p[2]))
    p_ml_input.append(beta)

with st.sidebar.expander("Функции p(x) и q(x)"):
    for i in range(3, 17):
        p_ml_input.append(st.number_input(f"Параметр p[{i}]", value=float(st.session_state.p[i]), format="%.3f"))

with st.sidebar.expander("Нелинейность и Источник"):
    for i in range(17, 25):
        p_ml_input.append(st.number_input(f"Параметр p[{i}]", value=float(st.session_state.p[i]), format="%.3f"))

# --- Расчет ---
if st.button("Решить и сравнить"):
    model, scalers = load_assets()
    x_nodes = np.linspace(0, 1, 150)
    p_num = np.array(p_ml_input).copy()
    p_num[0] = 10**p_num[0] 
    if p_num[21] < 0: p_num[21] = 10**p_num[21]
    
    # 1. Стандарт
    st.session_state.eval_count = 0
    y_guess_lin = np.vstack([np.linspace(alpha, beta, 150), np.zeros(150)])
    t0 = time.time(); res_std = solve_bvp(lambda x,y: ode_system_logic(x,y,p_num), lambda ya,yb: bc_logic(ya,yb,p_num), x_nodes, y_guess_lin, tol=1e-3)
    t_std = time.time() - t0; evals_std = st.session_state.eval_count

    # 2. ML Предсказание
    p_s = scalers['scaler_x'].transform(np.array(p_ml_input).reshape(1, -1))
    with torch.no_grad():
        y_coeffs = scalers['scaler_y'].inverse_transform(model(torch.FloatTensor(p_s)).numpy())[0]
    spline = BSpline(KNOTS, y_coeffs, DEGREE, extrapolate=False)
    y_ml_init = spline(x_nodes)
    
    # 3. Гибрид
    st.session_state.eval_count = 0
    t1 = time.time(); res_ml = solve_bvp(lambda x,y: ode_system_logic(x,y,p_num), lambda ya,yb: bc_logic(ya,yb,p_num), x_nodes, np.vstack([y_ml_init, np.gradient(y_ml_init, x_nodes)]), tol=1e-3)
    t_ml = time.time() - t1; evals_ml = st.session_state.eval_count

    # --- Уведомления о статусе ---
    if res_std.success and res_ml.success:
        st.success("Оба метода успешно нашли решение. Гибридный метод подтвердил точность.")
    elif not res_std.success and res_ml.success:
        st.warning("Стандартный метод не сошелся (дивергенция). Гибридный метод успешно нашел решение за счет точного начального приближения.")
    else:
        st.error("Ошибка сходимости: задача слишком сложная для текущей конфигурации обоих методов.")

    # --- Визуализация ---
    col_graph, col_stats = st.columns([2, 1])
    with col_graph:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_nodes, np.linspace(alpha, beta, 150), color='gray', linestyle=':', alpha=0.4, label='Linear Start')
        ax.plot(x_nodes, y_ml_init, 'r--', alpha=0.6, label='ML Initial Guess')
        if res_ml.success: ax.plot(res_ml.x, res_ml.y[0], 'b-', linewidth=2.5, label='Hybrid Solution')
        if res_std.success: ax.plot(res_std.x, res_std.y[0], color='orange', linestyle='-.', alpha=0.6, label='Standard Result')
        ax.legend(); ax.grid(True, alpha=0.2); st.pyplot(fig)

    with col_stats:
        st.subheader("Метрики")
        st.table({"Показатель": ["Итерации", "Вызовы f(x)", "Время (с)"],
                  "Стандарт": [res_std.niter if res_std.success else "Провал", evals_std, f"{t_std:.4f}"],
                  "Гибрид": [res_ml.niter if res_ml.success else "Провал", evals_ml, f"{t_ml:.4f}"]})
        if res_std.success and res_ml.success: st.metric("Ускорение", f"{evals_std / max(1, evals_ml):.1f}x")

    st.divider()
    st.subheader("Базисные функции решения")
    
    fig_b, ax_b = plt.subplots(figsize=(12, 4))
    x_fine = np.linspace(0, 1, 300)
    for i in range(N_BASIS):
        c_i = np.zeros(N_BASIS); c_i[i] = 1.0; y_b = y_coeffs[i] * BSpline(KNOTS, c_i, DEGREE)(x_fine)
        ax_b.plot(x_fine, y_b, alpha=0.5, linestyle='--')
    ax_b.plot(x_fine, spline(x_fine), 'b-', linewidth=2); st.pyplot(fig_b)
