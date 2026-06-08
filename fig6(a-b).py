import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt
#==================fig6(a)(b) computer code====================#
t2 = 0.5
dt = 0.1
total_time = 1000
def hamiltonian(t1, t2, L, gamma):
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
def evolution(H, psi0, dt, total_time):
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
gamma_value1 = np.arange(0, 1, 0.05)
gamma_value2 = np.arange(1, 10, 0.4)
gamma_value3 = np.arange(0, 0.11, 0.01)
gamma_value4 = np.arange(0.1, 1.1, 0.1)
gamma_value5 = np.arange(1, 31, 1)
t1 = 0.4
x0 = 90
L = 100
Pedge_04_1_100 = []
for gamma in gamma_value1:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_04_1_100.append(P[0])
Pedge_04_2_100 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_04_2_100.append(P[0])
t1 = 0.4
x0 = 135
L = 150
Pedge_04_1_150 = []
for gamma in gamma_value1:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_04_1_150.append(P[0])
Pedge_04_2_150 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_04_2_150.append(P[0])
t1 = 0.4
x0 = 270
L = 300
Pedge_04_1_300 = []
for gamma in gamma_value1:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_04_1_300.append(P[0])
Pedge_04_2_300 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_04_2_300.append(P[0])

t1 = 0.6
x0 = 90
L = 100
Pedge_06_1_100 = []
for gamma in gamma_value3:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_1_100.append(P[0])
Pedge_06_2_100 = []
for gamma in gamma_value4:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_2_100.append(P[0])
Pedge_06_3_100 = []
for gamma in gamma_value5:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_3_100.append(P[0])
t1 = 0.6
x0 = 135
L = 150
Pedge_06_1_150 = []
for gamma in gamma_value3:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_1_150.append(P[0])
Pedge_06_2_150 = []
for gamma in gamma_value4:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_2_150.append(P[0])
Pedge_06_3_150 = []
for gamma in gamma_value5:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_3_150.append(P[0])
t1 = 0.6
x0 = 270
L = 300
Pedge_06_1_300 = []
for gamma in gamma_value3:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_1_300.append(P[0])
Pedge_06_2_300 = []
for gamma in gamma_value4:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_2_300.append(P[0])
Pedge_06_3_300 = []
for gamma in gamma_value5:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_3_300.append(P[0])
fig = plt.figure(figsize=(9, 8))
gs_main = gridspec.GridSpec(1, 2, figure=fig)
ax_a = fig.add_subplot(gs_main[0, 0])
ax_a.plot(gamma_value1, Pedge_04_1_100, marker="o", markersize=4, label=r'$L=100$',color='k')
ax_a.plot(gamma_value2, Pedge_04_2_100, marker="o", markersize=4,color='k')
ax_a.plot(gamma_value1, Pedge_04_1_150, marker="x", markersize=6,label=r'$L=150$',color='red')
ax_a.plot(gamma_value2, Pedge_04_2_150, marker="x", markersize=6,color='red')
ax_a.plot(gamma_value1, Pedge_04_1_300, marker="+", markersize=6,label=r'$L=300$',color='blue')
ax_a.plot(gamma_value2, Pedge_04_2_300, marker="+", markersize=6,color='blue')
ax_a.set_xlim(-0.2, 10)
ax_a.set_xticks([0, 2, 4, 6, 8, 10])
ax_a.set_yticks([0,0.06,0.12])
ax_a.legend(frameon=False, fontsize=12, loc='lower right')
ax_a.set_xlabel(r'$\gamma$', size=20)
ax_a.set_ylabel(r'$P_1$', size=20)
ax_a.tick_params(labelsize=15)
gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 1], width_ratios=[2, 7], wspace=0.03)
ax_b_left = fig.add_subplot(gs_b[0])
ax_b_left.plot(gamma_value3, Pedge_06_1_100, marker="o",  markersize=4,color='k')
ax_b_left.plot(gamma_value4, Pedge_06_2_100, marker="o",  markersize=4,color='k')
ax_b_left.plot(gamma_value3, Pedge_06_1_150, marker="x",  markersize=6,color='red')
ax_b_left.plot(gamma_value4, Pedge_06_2_150, marker="x",  markersize=6,color='red')
ax_b_left.plot(gamma_value3, Pedge_06_1_300, marker="+",  markersize=6,color='blue')
ax_b_left.plot(gamma_value4, Pedge_06_2_300, marker="+",  markersize=6,color='blue')
ax_b_left.set_xlim(-0.1, 1)
ax_b_left.set_xticks([0, 1])
ax_b_left.set_yticks([0, 0.05])
ax_b_left.tick_params(labelsize=15)
ax_b_right = fig.add_subplot(gs_b[1], sharey=ax_b_left)
ax_b_right.plot(gamma_value5, Pedge_06_3_100, marker="o",  markersize=4,color='k',label=r'$L=100$')
ax_b_right.plot(gamma_value5, Pedge_06_3_150, marker="x",  markersize=6,color='red',label=r'$L=150$')
ax_b_right.plot(gamma_value5, Pedge_06_3_300, marker="+",  markersize=6,color='blue',label=r'$L=300$')
ax_b_right.tick_params(left=False, labelleft=False)
ax_b_right.set_xlim(1, 30)
ax_b_right.set_xticks([10, 20, 30])
plt.setp(ax_b_right.get_yticklabels(), visible=False)
ax_b_right.legend(frameon=False, fontsize=12, loc='lower right')
fig.text(0.77, 0.04, r'$\gamma$', fontsize=20)
ax_b_right.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig('kimi.png', dpi=400, bbox_inches='tight')
#==========To avoid duplicate calculations, data is stored in fig6 (a-b). npz============#
np.savez('fig6(a-b).npz',
         gamma_value1=gamma_value1,
         gamma_value2=gamma_value2,
         gamma_value3=gamma_value3,
         gamma_value4=gamma_value4,
         gamma_value5=gamma_value5,
         Pedge_04_1_100=Pedge_04_1_100,
         Pedge_04_2_100=Pedge_04_2_100,
         Pedge_04_1_150=Pedge_04_1_150,
         Pedge_04_2_150=Pedge_04_2_150,
         Pedge_04_1_300=Pedge_04_1_300,
         Pedge_04_2_300=Pedge_04_2_300,
         Pedge_06_1_100=Pedge_06_1_100,
         Pedge_06_2_100=Pedge_06_2_100,
         Pedge_06_3_100=Pedge_06_3_100,
         Pedge_06_1_150=Pedge_06_1_150,
         Pedge_06_2_150=Pedge_06_2_150,
         Pedge_06_3_150=Pedge_06_3_150,
         Pedge_06_1_300=Pedge_06_1_300,
         Pedge_06_2_300=Pedge_06_2_300,
         Pedge_06_3_300=Pedge_06_3_300,)
plt.show()
