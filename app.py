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

st.latex(r"""
\begin{cases} 
\varepsilon y'' + p(x)y' + q(x)y + j y^2 + k y^3 = f(x) \\
y(0) = \alpha, \quad y(1) = \beta
\end{cases}
""")
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
with st.sidebar.expander("Базовые параметры и ГУ", expanded=True):
    eps_val = st.number_input("Параметр ε", 0.0001, 0.1000, float(10**st.session_state.p[0]), format="%.4f")
    p_ml_input.append(np.log10(eps_val))
    p_ml_input.append(st.slider("alpha (y0)", -2.0, 2.0, float(st.session_state.p[1])))
    p_ml_input.append(st.slider("beta (y1)", -2.0, 2.0, float(st.session_state.p[2])))

with st.sidebar.expander("Коэффициенты p(x)"):
    p_ml_input.append(st.number_input("p0 (const)", value=float(st.session_state.p[3]), format="%.3f"))
    p_ml_input.append(st.number_input("p1 (x)", value=float(st.session_state.p[4]), format="%.3f"))
    p_ml_input.append(st.number_input("p2 (x^2)", value=float(st.session_state.p[5]), format="%.3f"))
    p_ml_input.append(st.number_input("w1 (sin amp)", value=float(st.session_state.p[6]), format="%.3f"))
    p_ml_input.append(st.number_input("w2 (sin freq)", value=float(st.session_state.p[7]), format="%.3f"))
    p_ml_input.append(st.number_input("v1 (cos amp)", value=float(st.session_state.p[8]), format="%.3f"))
    p_ml_input.append(st.number_input("v2 (cos freq)", value=float(st.session_state.p[9]), format="%.3f"))

with st.sidebar.expander("Коэффициенты q(x)"):
    p_ml_input.append(st.number_input("q0 (const)", value=float(st.session_state.p[10]), format="%.3f"))
    p_ml_input.append(st.number_input("q1 (x)", value=float(st.session_state.p[11]), format="%.3f"))
    p_ml_input.append(st.number_input("q2 (x^2)", value=float(st.session_state.p[12]), format="%.3f"))
    p_ml_input.append(st.number_input("e1 (sin amp)", value=float(st.session_state.p[13]), format="%.3f"))
    p_ml_input.append(st.number_input("e2 (sin freq)", value=float(st.session_state.p[14]), format="%.3f"))
    p_ml_input.append(st.number_input("u1 (cos amp)", value=float(st.session_state.p[15]), format="%.3f"))
    p_ml_input.append(st.number_input("u2 (cos freq)", value=float(st.session_state.p[16]), format="%.3f"))

with st.sidebar.expander("Нелинейность и Источник"):
    p_ml_input.append(st.number_input("j (коэф. y^2)", value=float(st.session_state.p[17]), format="%.3f"))
    p_ml_input.append(st.number_input("k (коэф. y^3)", value=float(st.session_state.p[18]), format="%.3f"))
    p_ml_input.append(st.number_input("A (амплитуда f)", value=float(st.session_state.p[19]), format="%.3f"))
    p_ml_input.append(st.number_input("mu (центр пика)", value=float(st.session_state.p[20]), format="%.3f"))
    p_ml_input.append(st.number_input("log10 sigma (ширина)", value=float(st.session_state.p[21]), format="%.3f"))
    p_ml_input.append(st.number_input("c0 (const f)", value=float(st.session_state.p[22]), format="%.3f"))
    p_ml_input.append(st.number_input("c1 (x f)", value=float(st.session_state.p[23]), format="%.3f"))
    p_ml_input.append(st.number_input("c2 (x^2 f)", value=float(st.session_state.p[24]), format="%.3f"))

# --- Расчет ---
if st.button("Решить и сравнить"):
    model, scalers = load_assets()
    x_nodes = np.linspace(0, 1, 500)
    p_num = np.array(p_ml_input).copy()
    p_num[0] = 10**p_num[0] 
    if p_num[21] < 0: p_num[21] = 10**p_num[21]
    
    # 1. Стандарт
    st.session_state.eval_count = 0
    y_guess_lin = np.vstack([np.linspace(p_ml_input[1], p_ml_input[2], len(x_nodes)), np.zeros(len(x_nodes))])
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

    # --- Уведомления ---
    if res_std.success and res_ml.success:
        st.success("Оба метода успешно нашли решение.")
    elif not res_std.success and res_ml.success:
        st.warning("Стандартный метод провалился. Гибрид нашел решение.")
    else:
        st.error("Критическая сложность: сходимость не достигнута.")

    # --- ГРАФИК РЕШЕНИЯ ---
    col_graph, col_stats = st.columns([2, 1])
    with col_graph:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_nodes, np.linspace(p_ml_input[1], p_ml_input[2], len(x_nodes)), color='gray', linestyle=':', alpha=0.4, label='Linear Start')
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

    # --- АНАЛИЗ БАЗИСА ---
    st.divider()
    st.subheader("Разложение решения по B-сплайнам")
    st.latex(r"y_{pred}(x) = \sum_{i=1}^{16} c_i \cdot B_{i,3}(x)")
    
    
    
    fig_b, ax_b = plt.subplots(figsize=(12, 4))
    x_fine = np.linspace(0, 1, 400)
    for i in range(N_BASIS):
        c_i_basis = np.zeros(N_BASIS); c_i_basis[i] = 1.0
        y_b = y_coeffs[i] * BSpline(KNOTS, c_i_basis, DEGREE)(x_fine)
        ax_b.plot(x_fine, y_b, alpha=0.5, linestyle='--', label=f'c_{i+1}' if i < 3 else None)
    ax_b.plot(x_fine, spline(x_fine), 'b-', linewidth=2.5, label='Resulting Spline')
    ax_b.grid(True, alpha=0.15); st.pyplot(fig_b)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Веса сплайнов $c_i$:")
        st.json({f"c_{i+1}": float(val) for i, val in enumerate(y_coeffs)})
    with c2:
        st.write("Входные параметры:")
        param_names = ["log10(eps)", "alpha", "beta", "p0", "p1", "p2", "w1", "w2", "v1", "v2", "q0", "q1", "q2", "e1", "e2", "u1", "u2", "j", "k", "A", "mu", "log10(sigma)", "c0", "c1", "c2"]
        st.json({name: float(val) for name, val in zip(param_names, p_ml_input)})
