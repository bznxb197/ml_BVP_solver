import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from generate_dataset_V4 import build_bspline_basis # Должен быть в той же папке

# =================================================================
# 1. ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ (В точности как при обучении Turbo)
# =================================================================
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

# =================================================================
# 2. КЛАСС-РЕШАТЕЛЬ (ГИБРИДНЫЙ ПОДХОД)
# =================================================================
class DeepBVP_Solver:
    def __init__(self, model_path, scalers_path):
        print(f"Загрузка компонентов из {model_path} и {scalers_path}...")
        
        # Загрузка скалеров
        self.scalers = joblib.load(scalers_path)
        self.scaler_x = self.scalers['scaler_x']
        self.scaler_y = self.scalers['scaler_y']
        
        # Инициализация и загрузка весов модели
        self.model = BVPNetTurbo()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Сетка и базис для генерации начального приближения
        self.x_nodes = np.linspace(0, 1, 500)
        self.Phi = build_bspline_basis(self.x_nodes)
        
    def get_nn_guess(self, params):
        """
        Превращает физические параметры в начальное приближение через NN.
        """
        # Модель обучена на векторе из 25 параметров
        full_params = np.zeros(25)
        full_params[0] = np.log10(params[0]) # eps в логарифмической шкале
        full_params[1] = params[1]           # граничное условие слева (alpha)
        full_params[2] = params[2]           # граничное условие справа (beta)
        
        # Нормализация и инференс
        p_s = self.scaler_x.transform(full_params.reshape(1, -1))
        with torch.no_grad():
            y_coeffs_s = self.model(torch.FloatTensor(p_s)).numpy()
            y_coeffs = self.scaler_y.inverse_transform(y_coeffs_s)[0]
        
        # Восстановление функции y(x) и её градиента
        y_pred = self.Phi @ y_coeffs
        y_prime = np.gradient(y_pred, self.x_nodes)
        return np.vstack([y_pred, y_prime])

# =================================================================
# 3. ФИЗИЧЕСКАЯ ПОСТАНОВКА: УРАВНЕНИЕ БЛАЗИУСА
# =================================================================
def blasius_ode(x, y, p):
    # Упрощенное уравнение погранслоя: f'' = -f * f' / eps
    eps = p[0]
    return np.vstack([y[1], -y[0] * y[1] / eps])

def blasius_bc(ya, yb, p):
    # Граничные условия: f(0)=alpha, f(1)=beta
    return np.array([ya[0] - p[1], yb[0] - p[2]])

# =================================================================
# 4. ЗАПУСК РАСЧЕТА И ВИЗУАЛИЗАЦИЯ
# =================================================================
if __name__ == "__main__":
    MODEL_PATH = "model_nn_v5_turbo.pth"
    SCALER_PATH = "scalers_v5.pkl"
    
    try:
        solver = DeepBVP_Solver(MODEL_PATH, SCALER_PATH)
        
        # Экстремально жесткий случай: eps = 0.0005 (очень тонкий погранслой)
        test_params = [0.005, 0.0, 1.0] 
        
        print(f"Генерация старта для eps={test_params[0]}...")
        guess_dl = solver.get_nn_guess(test_params)
        
        print("Запуск численного решателя...")
        sol = solve_bvp(lambda x,y: blasius_ode(x,y,test_params), 
                        lambda ya,yb: blasius_bc(ya,yb,test_params), 
                        solver.x_nodes, guess_dl, tol=1e-3)
        
        if sol.success:
            print(f"РЕШЕНО! Итераций: {sol.niter}")
            
            

            # Отрисовка
            plt.figure(figsize=(10, 6))
            plt.plot(solver.x_nodes, guess_dl[0], 'r--', alpha=0.5, label='Нейросеть (Старт)')
            plt.plot(sol.x, sol.y[0], 'b-', linewidth=2, label='SciPy (Итог)')
            plt.fill_between(sol.x, sol.y[0], color='blue', alpha=0.1)
            
            plt.title(f"Решение уравнения Блазиуса (eps={test_params[0]})", fontsize=14)
            plt.xlabel("Координата x", fontsize=12)
            plt.ylabel("Скорость потока f(x)", fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.show()
        else:
            print("Ошибка: Численный метод не смог сойтись даже с подсказкой NN.")
            
    except Exception as e:
        print(f"Критическая ошибка: {e}")
