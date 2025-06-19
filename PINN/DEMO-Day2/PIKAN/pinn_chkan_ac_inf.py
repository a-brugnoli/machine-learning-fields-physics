from kan import KAN, LBFGS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from ChebyKANLayer import ChebyKANLayer
import time
import scipy.io as sio


pi = math.pi
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ChebyKAN(nn.Module):
    def __init__(self):
        super(ChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(2, 32, 5)
        self.chebykan2 = ChebyKANLayer(32, 32, 5)
        self.chebykan3 = ChebyKANLayer(32, 1, 5)

    def forward(self, x):
        x = self.chebykan1(x)
        x = self.chebykan2(x)
        x = self.chebykan3(x)
        return x

def ac_loss(model, x, t, D):
    x.requires_grad = True
    t.requires_grad = True
    u = model(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    f = u_t - D * u_xx + 5.0 * (u**3 - u)
    return f


def data_loss(model, x, t, u_true):
    u_pred = model(torch.cat([x, t], dim=1))
    return (u_pred - u_true).pow(2).mean()

layers = [2, 16, 16,1]

model = ChebyKAN().to(device)

optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
model.load_state_dict(torch.load('ac_model.pth'))

optimizer = optim.Adam(model.parameters(), lr=5e-4)
data = sio.loadmat("AC.mat")

xx = data["x"]
tt = data["tt"]
u_star = data["uu"]
u_star = u_star.reshape(-1, 1)

xx = xx.reshape(-1, 1)
tt = tt.reshape(-1, 1)
xx, tt = np.meshgrid(xx, tt)
xx = xx.T
tt = tt.T
np.save("xx.npy", xx,  allow_pickle=True)
np.save("tt.npy", tt, allow_pickle=True)
print(f"Shape of xx: {xx.shape} and Shape of tt: {tt.shape}")

xx = xx.reshape(-1, 1)
tt = tt.reshape(-1, 1)
xx_tensor = torch.tensor(xx, dtype=torch.float32).to(device)
tt_tensor = torch.tensor(tt, dtype=torch.float32).to(device)

with torch.no_grad():
    u_pred = model(torch.cat([xx_tensor, tt_tensor], dim=1))

u_pred = u_pred.cpu().numpy()
e_abs = u_pred - u_star

np.save("u_pred.npy", u_pred, allow_pickle=True)
np.save("u_star.npy", u_star, allow_pickle=True)
np.save("u_err.npy",  e_abs, allow_pickle=True)
u_err = np.linalg.norm(u_star - u_pred)/np.linalg.norm(u_star)
print(f"u_err: {u_err * 100.0} %")
