from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib 
import numpy as np
from model_architecture import BVPNetTurbo
import torch

app = FastAPI()
def load_model(model_path="model_nn_v5_turbo.pth", scaler_path="scalers_v5.pkl"):

    scalers = joblib.load(scaler_path)
    scaler_x = scalers['scaler_x']
    scaler_y = scalers['scaler_y']
    

    model = BVPNetTurbo(input_dim=25, output_dim=16, hidden=256)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, scaler_x, scaler_y

MODEL, SCALER_X, SCALER_Y = load_model()

class InputData(BaseModel):
    eps: float = Field(..., ge=1e-3, le=0.1, description="epsilon parameter")
    alpha: float = Field(..., ge=-2, le=2, description="alpha parameter")
    beta: float = Field(..., ge=-2, le=2, description="beta parameter")
    p0: float = Field(..., ge=-3, le=3, description="p0 coefficient")
    p1: float = Field(..., ge=-3, le=3, description="p1 coefficient")
    p2: float = Field(..., ge=-3, le=3, description="p2 coefficient")
    w1: float = Field(..., ge=-1.5, le=1.5, description="w1 parameter")
    w2: float = Field(..., ge=1, le=5.0119, description="w2 parameter (log-uniform 10^0 to 10^0.7)")
    v1: float = Field(..., ge=-1.5, le=1.5, description="v1 parameter")
    v2: float = Field(..., ge=1, le=5.0119, description="v2 parameter (log-uniform 10^0 to 10^0.7)")
    q0: float = Field(..., ge=-3, le=3, description="q0 coefficient")
    q1: float = Field(..., ge=-3, le=3, description="q1 coefficient")
    q2: float = Field(..., ge=-3, le=3, description="q2 coefficient")
    e1: float = Field(..., ge=-1.5, le=1.5, description="e1 parameter")
    e2: float = Field(..., ge=1, le=5.0119, description="e2 parameter (log-uniform 10^0 to 10^0.7)")
    u1: float = Field(..., ge=-1.5, le=1.5, description="u1 parameter")
    u2: float = Field(..., ge=1, le=5.0119, description="u2 parameter (log-uniform 10^0 to 10^0.7)")
    j: float = Field(..., ge=-0.8, le=0.8, description="j parameter")
    k: float = Field(..., ge=-0.8, le=0.8, description="k parameter")
    A: float = Field(..., ge=-4, le=4, description="A amplitude parameter")
    mu: float = Field(..., ge=0.2, le=0.8, description="mu parameter")
    sigma: float = Field(..., ge=0.01, le=0.3162, description="sigma parameter (log-uniform 10^-2 to 10^-0.5)")
    c0: float = Field(..., ge=-3, le=3, description="c0 coefficient")
    c1: float = Field(..., ge=-3, le=3, description="c1 coefficient")
    c2: float = Field(..., ge=-3, le=3, description="c2 coefficient")
    

class PredictionResponse(BaseModel):
    basis_00: float = Field(..., description="Coefficient 0 of cubic B-spline basis")
    basis_01: float = Field(..., description="Coefficient 1 of cubic B-spline basis")
    basis_02: float = Field(..., description="Coefficient 2 of cubic B-spline basis")
    basis_03: float = Field(..., description="Coefficient 3 of cubic B-spline basis")
    basis_04: float = Field(..., description="Coefficient 4 of cubic B-spline basis")
    basis_05: float = Field(..., description="Coefficient 5 of cubic B-spline basis")
    basis_06: float = Field(..., description="Coefficient 6 of cubic B-spline basis")
    basis_07: float = Field(..., description="Coefficient 7 of cubic B-spline basis")
    basis_08: float = Field(..., description="Coefficient 8 of cubic B-spline basis")
    basis_09: float = Field(..., description="Coefficient 9 of cubic B-spline basis")
    basis_10: float = Field(..., description="Coefficient 10 of cubic B-spline basis")
    basis_11: float = Field(..., description="Coefficient 11 of cubic B-spline basis")
    basis_12: float = Field(..., description="Coefficient 12 of cubic B-spline basis")
    basis_13: float = Field(..., description="Coefficient 13 of cubic B-spline basis")
    basis_14: float = Field(..., description="Coefficient 14 of cubic B-spline basis")
    basis_15: float = Field(..., description="Coefficient 15 of cubic B-spline basis")



@app.post("/predict")
def predict(data: InputData):
    # Преобразуем входные данные в массив
    features = np.array([
        data.eps,
        data.alpha,
        data.beta,
        data.p0,
        data.p1,
        data.p2,
        data.w1,
        data.w2,
        data.v1,
        data.v2,
        data.q0,
        data.q1,
        data.q2,
        data.e1,
        data.e2,
        data.u1,
        data.u2,
        data.j,
        data.k,
        data.A,
        data.mu,
        data.sigma,
        data.c0,
        data.c1,
        data.c2
    ]).reshape(1, -1)

    features_scaled = SCALER_X.transform(features)
    
    features_tensor = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        prediction_scaled = MODEL(features_tensor).numpy()

    prediction = SCALER_Y.inverse_transform(prediction_scaled)

    return PredictionResponse(
        basis_00=prediction[0][0],
        basis_01=prediction[0][1],
        basis_02=prediction[0][2],
        basis_03=prediction[0][3],
        basis_04=prediction[0][4],
        basis_05=prediction[0][5],
        basis_06=prediction[0][6],
        basis_07=prediction[0][7],
        basis_08=prediction[0][8],
        basis_09=prediction[0][9],
        basis_10=prediction[0][10],
        basis_11=prediction[0][11],
        basis_12=prediction[0][12],
        basis_13=prediction[0][13],
        basis_14=prediction[0][14],
        basis_15=prediction[0][15]
    )