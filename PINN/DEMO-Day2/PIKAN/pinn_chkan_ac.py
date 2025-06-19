from kan import KAN, LBFGS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pyDOE import lhs
import math
from ChebyKANLayer import ChebyKANLayer
import time

pi = math.pi
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ChebyKAN(nn.Module):
    def __init__(self):
        super(ChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(2, 128, 5)
        self.chebykan2 = ChebyKANLayer(128, 128, 5)
        self.chebykan3 = ChebyKANLayer(128, 128, 5)
        self.chebykan4 = ChebyKANLayer(128, 128, 5)
        self.chebykan5 = ChebyKANLayer(128, 1, 5)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.chebykan5(self.act(self.chebykan4(self.act(self.chebykan3(self.act(self.chebykan2(self.act(self.chebykan1(x)))))))))
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

#model = KAN(width=[2,16,16,1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25, device=device)

model = ChebyKAN().to(device)

optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)


#model.load_state_dict(torch.load('ac_model.pth'))

optimizer = optim.Adam(model.parameters(), lr=5e-4)
D = 0.0001
N_i = 200
N_b = 100
N_f = 50000
lb = np.array([-1, 0])
ub = np.array([1, 1])

xt_f = lb + (ub - lb)*lhs(2, N_f) 

#### TODo Fill in Right Valies for IC and BCs
x_ic = torch.linspace(-1, 1, steps=N_i).view(-1, 1).to(device)
t_ic = torch.zeros(N_i, 1).view(-1, 1).to(device)
u_ic = (x_ic**2 * torch.cos(pi*x_ic)).view(-1, 1).to(device)

x_lb = -1.0 * torch.ones(N_b, 1).view(-1, 1).to(device)
t_lb = torch.linspace(0, 1, steps=N_b).view(-1, 1).to(device)
u_lb = -1.0 * torch.ones(N_b, 1).view(-1, 1).to(device)

x_rb = 1.0 * torch.ones(N_b, 1).view(-1, 1).to(device)
t_rb = torch.linspace(0, 1, steps=N_b).view(-1, 1).to(device)
u_rb = -1.0 * torch.ones(N_b, 1).to(device)

x_f = torch.tensor(xt_f[:, 0], dtype=torch.float32).view(-1, 1).to(device)
t_f = torch.tensor(xt_f[:, 1], dtype=torch.float32).view(-1, 1).to(device)

print(f"x_f: {x_f}")
print(f"t_f: {t_f}")

loss_list = []

t1 = time.time()
num_epochs = 150001
for epoch in range(num_epochs):
    optimizer.zero_grad()
    f_i = ac_loss(model, x_f, t_f, 1e-04)
    f_loss = torch.mean((f_i)**2)
    l_b_loss = data_loss(model, x_lb, t_lb, u_lb)
    r_b_loss = data_loss(model, x_rb, t_rb, u_rb)
    i_c_loss = data_loss(model, x_ic, t_ic, u_ic)
    loss = f_loss + l_b_loss + r_b_loss + i_c_loss
    loss.backward()
    optimizer.step()
    t2 = time.time()
    loss_list.append(loss.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, f_loss: {f_loss.item()}, total_loss: {loss.item()} and time: {t2-t1} s')


t3 = time.time()
print(f"Elapsed Time: {t3-t1} sec")
#### validation loss
N_val = 200
x_val = np.linspace(-1, 1, N_val, dtype=float).reshape(-1, 1)
t_val = np.linspace(0, 1, N_val, dtype=float).reshape(-1, 1)

xx, tt = np.meshgrid(x_val, t_val)

xx_cpu = xx.reshape(-1, 1)
tt_cpu = tt.reshape(-1, 1)

xx = torch.tensor(xx_cpu, dtype=torch.float32).to(device)
tt = torch.tensor(tt_cpu, dtype=torch.float32).to(device)

xx_cpu = xx_cpu.reshape(N_val, N_val)
tt_cpu = tt_cpu.reshape(N_val, N_val)

torch.save(model.state_dict(), 'ac_model.pth')
np.save("loss_hist.npy", loss_list, allow_pickle=True)

with torch.no_grad():
    u_pred = model(torch.cat([xx, tt], dim=1))

u_pred = u_pred.cpu()
u_pred = u_pred.reshape(N_val, N_val)
plt.imshow(u_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[tt_cpu.min(), tt_cpu.max(), xx_cpu.min(), xx_cpu.max()], 
                  origin='lower', aspect='auto')

plt.legend()
plt.savefig("ac_sol.png")
