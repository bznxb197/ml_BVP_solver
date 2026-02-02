import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp
from scipy.interpolate import BSpline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Настройки базиса
DEGREE = 3
N_BASIS = 16 
knots_internal = np.linspace(0, 1, N_BASIS - DEGREE + 1)
KNOTS = np.concatenate((np.zeros(DEGREE), knots_internal, np.ones(DEGREE)))

def build_bspline_basis(x):
    Phi = np.zeros((len(x), N_BASIS))
    for i in range(N_BASIS):
        c = np.zeros(N_BASIS)
        c[i] = 1.0
        Phi[:, i] = BSpline(KNOTS, c, DEGREE, extrapolate=False)(x)
    return Phi

def get_params():
    """Генерирует ровно 25 параметров ОДУ"""
    p = [
        10**np.random.uniform(np.log10(1e-4), np.log10(0.05)), # eps (0)
        np.random.uniform(-2, 2), # alpha (1)
        np.random.uniform(-2, 2), # beta (2)
        *np.random.uniform(-3, 3, size=3), # p0, p1, p2 (3,4,5)
        np.random.uniform(-1.5, 1.5),      # w1 (6)
        10**np.random.uniform(0, 0.7),     # w2 (7)
        np.random.uniform(-1.5, 1.5),      # v1 (8)
        10**np.random.uniform(0, 0.7),     # v2 (9)
        *np.random.uniform(-3, 3, size=3), # q0, q1, q2 (10,11,12)
        np.random.uniform(-1.5, 1.5),      # e1 (13)
        10**np.random.uniform(0, 0.7),     # e2 (14)
        np.random.uniform(-1.5, 1.5),      # u1 (15)
        10**np.random.uniform(0, 0.7),     # u2 (16)
        np.random.uniform(-2.0, 2.0),      # j (17)
        np.random.uniform(-1.0, 1.0),      # k (18)
        np.random.uniform(-4, 4),          # A (19)
        np.random.uniform(0.2, 0.8),       # mu (20)
        10**np.random.uniform(-2.0, -0.5), # sigma (21)
        *np.random.uniform(-3, 3, size=3)  # c0, c1, c2 (22,23,24)
    ]
    return np.array(p)

def ode_system(x, y, p):
    eps = p[0]
    # Собираем функции p(x), q(x) и f(x)
    px = p[3] + p[4]*x + p[5]*x**2 + p[6]*np.sin(p[7]*x) + p[8]*np.cos(p[9]*x)
    qx = p[10] + p[11]*x + p[12]*x**2 + p[13]*np.sin(p[14]*x) + p[15]*np.cos(p[16]*x)
    fx = p[19]*np.exp(-((x - p[20])/p[21])**2) + p[22] + p[23]*x + p[24]*x**2
    
    return np.vstack([
        y[1],
        (fx - px*y[1] - qx*y[0] - p[17]*y[0]**2 - p[18]*y[0]**3) / eps
    ])

def bc(ya, yb, p):
    return np.array([ya[0] - p[1], yb[0] - p[2]])

def run_generation(N=15000, filename="dataset_V4.csv"):
    x_nodes = np.linspace(0, 1, 150)
    x_fit = np.linspace(0, 1, 200)
    Phi = build_bspline_basis(x_fit)
    
    data = []
    for _ in tqdm(range(N), desc="Solving"):
        p = get_params()
        # Начальное приближение - прямая
        y_guess = np.vstack([np.linspace(p[1], p[2], len(x_nodes)), np.zeros(len(x_nodes))])
        
        try:
            sol = solve_bvp(lambda x,y: ode_system(x,y,p), 
                            lambda ya,yb: bc(ya,yb,p), 
                            x_nodes, y_guess, tol=1e-3)
            if sol.success:
                y_fine = sol.sol(x_fit)[0]
                coeffs, *_ = np.linalg.lstsq(Phi, y_fine, rcond=None)
                
                # Трансформируем признаки для ML: eps -> log10(eps)
                p_ml = p.copy()
                p_ml[0] = np.log10(p[0])
                data.append(np.hstack([p_ml, coeffs]))
        except:
            continue
            
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\nSaved {len(df)} samples.")

if __name__ == "__main__":
    run_generation(40000)
