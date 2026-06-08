import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from matplotlib.lines import Line2D
#===================fig 7 code==================#
def hamiltonian(t1, t2, gamma, L):
    H = np.zeros((2 * L, 2 * L), dtype=complex)
    for i in np.arange(0, 2 * L, 2):
        H[i, i + 1] = t1
    for i in np.arange(1, 2 * L - 1, 2):
        H[i, i + 1] = t2 / 2
    for i in np.arange(0, 2 * L - 2, 2):
        H[i, i + 2] = -1j * t2 / 2
    for i in np.arange(1, 2 * L - 1, 2):
        H[i, i + 2] = 1j * t2 / 2
    for i in np.arange(0, 2 * L - 2, 2):
        H[i, i + 3] = t2 / 2
    for i in np.arange(0, 2 * L, 2):
        H[i + 1, i] = t1
    for i in np.arange(1, 2 * L - 1, 2):
        H[i + 1, i] = t2 / 2
    for i in np.arange(0, 2 * L - 2, 2):
        H[i + 2, i] = 1j * t2 / 2
    for i in np.arange(1, 2 * L - 1, 2):
        H[i + 2, i] = -1j * t2 / 2
    for i in np.arange(0, 2 * L - 2, 2):
        H[i + 3, i] = t2 / 2
    for i in np.arange(1, 2 * L, 2):
        H[i, i] = -1j * gamma
    return H
def initial_state(L, x0):
    psi = np.zeros(2 * L, dtype=complex)
    psi[2 * x0 - 2] = 1
    return psi
def time_evolution(H, psi0, dt, total_time):
    def schrodinger(t, psi):
        return -1j * H @ psi
    t_span = [0, total_time]
    t_eval = np.arange(0, total_time, dt)
    sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval, method='RK45')
    return sol.y
def loss_probability(psi, gamma, L, dt):
    P = np.zeros(L)
    for x in range(L):
        P[x] = 2 * gamma * trapezoid(np.abs(psi[2 * x + 1, :]) ** 2, dx=dt)
    return P
file_path = r'C:\Users\26365\Desktop\figure\Image related code\fig7\fig7a.npy'
all_data = np.load(file_path, allow_pickle=True).item()
gamma_values = [0.8, 2, 5, 10]
colors = ['blue', 'red', 'black', 'green']
labels = [r'$\gamma=0.8$', r'$\gamma=2$', r'$\gamma=5$', r'$\gamma=10$']
x0_values = np.arange(100, 205, 5)
fig, (ax_top, ax_bottom) = plt.subplots(1, 2,figsize=(9.4, 4))
for gamma, color, label in zip(gamma_values, colors, labels):
    ratio_values = all_data[gamma]['ratio']
    ax_top.plot(x0_values, ratio_values, 'o-', color=color, markersize=3,
                linewidth=1.5, label=label)
x_ref = np.array([110, 190])
ax_top.plot(x_ref, 2 * x_ref - 30, 'b--', linewidth=1.5, label=r'k=2')
ax_top.plot(x_ref, 1.4 * x_ref - 16, 'r--', linewidth=1.5, label=r'k=1.4')
ax_top.plot(x_ref, 0.54 * x_ref + 47, 'k--', linewidth=1.5, label=r'k=0.54')
ax_top.plot(x_ref, 0.43 * x_ref + 32, 'g--', linewidth=1.5, label=r'k=0.43')
ax_top.set_xticks([100, 150, 200])
ax_top.set_xlabel(r'$x_0$',size=15)
ax_top.set_yticks([100, 200, 300])
ax_top.set_ylabel(r'$P_1/P_{\min}$',size=12)
handles = [
    Line2D([0], [0], color='blue', marker='o', linestyle='-', markersize=2, label=r'$\gamma=0.8$'),
    Line2D([0], [0], color='red', marker='o', linestyle='-', markersize=2, label=r'$\gamma=2$'),
    Line2D([0], [0], color='black', marker='o', linestyle='-', markersize=2, label=r'$\gamma=5$'),
    Line2D([0], [0], color='green', marker='o', linestyle='-', markersize=2, label=r'$\gamma=10$'),
    Line2D([0], [0], color='blue', linestyle='--', label='k=2'),
    Line2D([0], [0], color='red', linestyle='--', label='k=1.4'),
    Line2D([0], [0], color='black', linestyle='--', label='k=0.54'),
    Line2D([0], [0], color='green', linestyle='--', label='k=0.43'),]
ax_top.legend(handles=handles, ncol=2, loc='upper left', fontsize=8.4, frameon=False)

x0 = 450
t1 = 0.6
t2 = 0.5
L = 500
dt = 0.1
total_time = 1000
gamma = 5
xx0_values = np.arange(10, 402, 2)
x0_values_ref = np.arange(10, 410, 10)
Pbulk_1 = []
H = hamiltonian(t1, t2, gamma, L)
psi0 = initial_state(L, x0)
psi_t = time_evolution(H, psi0, dt, total_time)
for xx0 in xx0_values:
    P = loss_probability(psi_t, gamma, L, dt)
    P_bulk = P[abs(xx0 - x0) - 1] / P[439]
    Pbulk_1.append(P_bulk)
ax_bottom.plot(xx0_values, Pbulk_1, "r-o", markersize=2)
ax_bottom.plot(x0_values_ref, 10 / (x0_values_ref), "k--", linewidth=1.8, label=r"$\sim |x-x_0|^{-1}$")
ax_bottom.plot(x0_values_ref, 0.07 * 0.992 ** x0_values_ref, "k^", markersize=5,
               label=r"$\sim (\min|\beta_L(\omega)|^{-2})^{|x-x_0|}$")
ax_bottom.set_yscale("log")
ax_bottom.set_xlabel(r"$|x-x_0|$",size=15)
ax_bottom.set_ylabel(r"$P_x/ P_{440}$",size=12)
ax_bottom.set_xticks([0, 100, 200, 300, 400])
ax_bottom.set_xlim(0,410)
ax_bottom.legend(fontsize=11,frameon=False)
fig.text(0.1, 0.89, '(a)', fontsize=12, transform=fig.transFigure)
fig.text(0.52, 0.89, '(b)', fontsize=12, transform=fig.transFigure)
plt.savefig('fig7.png', dpi=400, bbox_inches='tight')
plt.show()