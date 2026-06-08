import matplotlib.gridspec as gridspec
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt
#============Numerical calculation of edge escape probability and PBC spectrum========#
x0 = 90
L = 100
t2 = 0.5
dt = 0.1
total_time = 800
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
def pauli_matrix(n):
    if n == 1:
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif n == 2:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif n == 3:
        return np.array([[1, 0], [0, -1]], dtype=complex)
    else:
        return np.eye(2, dtype=complex)
def h(k):
    return (t1 + t2 * np.cos(k)) * pauli_matrix(1) + (t2 * np.sin(k) + 1j * r / 2) * pauli_matrix(3) - 1j * r / 2 * np.eye(2)
k_values = np.arange(0, 2 * np.pi, 0.005)
gamma_value1 = np.arange(0, 1, 0.05)
gamma_value2 = np.arange(1, 10, 0.4)
gamma_value3 = np.arange(0, 0.11, 0.01)
gamma_value4 = np.arange(0.1, 1.1, 0.1)
gamma_value5 = np.arange(1, 31, 1)
#===================imaginary gap opening====================#
t1 = 0.2
Pedge_02_1 = []
for gamma in gamma_value1:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_02_1.append(P[0])
Pedge_02_2 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_02_2.append(P[0])
t1 = 0.3
Pedge_03_1 = []
for gamma in gamma_value1:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_03_1.append(P[0])
Pedge_03_2 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_03_2.append(P[0])
t1 = 0.4
Pedge_04_1 = []
for gamma in gamma_value1:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_04_1.append(P[0])
Pedge_04_2 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_04_2.append(P[0])
#==========================imaginary gap closing================#
t1 = 0.6
Pedge_06_1 = []
for gamma in gamma_value3:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_1.append(P[0])
Pedge_06_2 = []
for gamma in gamma_value4:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_2.append(P[0])
Pedge_06_3 = []
for gamma in gamma_value5:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_06_3.append(P[0])
t1 = 0.7
Pedge_07_1 = []
for gamma in gamma_value3:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_07_1.append(P[0])
Pedge_07_2 = []
for gamma in gamma_value4:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_07_2.append(P[0])
Pedge_07_3 = []
for gamma in gamma_value5:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_07_3.append(P[0])
t1 = 0.8
Pedge_08_1 = []
for gamma in gamma_value3:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_08_1.append(P[0])
Pedge_08_2 = []
for gamma in gamma_value4:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_08_2.append(P[0])
Pedge_08_3 = []
for gamma in gamma_value5:
    H = hamiltonian(t1, t2, L, gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge_08_3.append(P[0])
#=============================PBC spectrum=================#
t1, t2, r = 0.2, 0.5, 0.5
t22, t222 = [], []
for k in k_values:
    eigenvalues = np.linalg.eigvals(h(k))
    t22.append(eigenvalues[0])
    t222.append(eigenvalues[1])
t1, t2, r = 0.3, 0.5, 0.5
t33, t333 = [], []
for k in k_values:
    eigenvalues = np.linalg.eigvals(h(k))
    t33.append(eigenvalues[0])
    t333.append(eigenvalues[1])
t1, t2, r = 0.4, 0.5, 0.5
t44, t444 = [], []
for k in k_values:
    eigenvalues = np.linalg.eigvals(h(k))
    t44.append(eigenvalues[0])
    t444.append(eigenvalues[1])
t1, t2, r = 0.6, 0.5, 0.5
t66, t666 = [], []
for k in k_values:
    eigenvalues = np.linalg.eigvals(h(k))
    t66.append(eigenvalues[0])
    t666.append(eigenvalues[1])
t1, t2, r = 0.7, 0.5, 0.5
t77, t777 = [], []
for k in k_values:
    eigenvalues = np.linalg.eigvals(h(k))
    t77.append(eigenvalues[0])
    t777.append(eigenvalues[1])
t1, t2, r = 0.8, 0.5, 0.5
t88, t888 = [], []
for k in k_values:
    eigenvalues = np.linalg.eigvals(h(k))
    t88.append(eigenvalues[0])
    t888.append(eigenvalues[1])
#===============================================drawing===================================#
fig = plt.figure()
gs_main = gridspec.GridSpec(2, 2, figure=fig)
ax_a = fig.add_subplot(gs_main[0, 0])
ax_a.plot(gamma_value1, Pedge_02_1, "r", marker="o", markersize=5, label=r'$t_1=0.2$')
ax_a.plot(gamma_value2, Pedge_02_2, "r", marker="o", markersize=5)
ax_a.plot(gamma_value1, Pedge_03_1, "b", marker="o", markersize=5, label=r'$t_1=0.3$')
ax_a.plot(gamma_value2, Pedge_03_2, "b", marker="o", markersize=5)
ax_a.plot(gamma_value1, Pedge_04_1, "k", marker="o", markersize=5, label=r'$t_1=0.4$')
ax_a.plot(gamma_value2, Pedge_04_2, "k", marker="o", markersize=5)
ax_a.set_xlim(-0.2, 10)
ax_a.set_xticks([0, 2, 4, 6, 8, 10])
ax_a.set_ylim(0, 0.14)
ax_a.set_yticks([0, 0.06,0.12])
ax_a.legend(frameon=False, fontsize=12, loc='upper right')
ax_a.set_xlabel(r'$\gamma$', size=15, weight='bold')
ax_a.set_ylabel(r'$P_1$', size=15, weight='bold')
ax_a.tick_params(labelsize=15)

gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 1], width_ratios=[2, 7], wspace=0.03)
ax_b_left = fig.add_subplot(gs_b[0])
ax_b_left.plot(gamma_value3, Pedge_06_1, "green", marker="o",  markersize=5)
ax_b_left.plot(gamma_value3, Pedge_07_1, "orange", marker="o", markersize=5)
ax_b_left.plot(gamma_value3, Pedge_08_1, "gray", marker="o",   markersize=5)
ax_b_left.plot(gamma_value4, Pedge_06_2, "green", marker="o",  markersize=5)
ax_b_left.plot(gamma_value4, Pedge_07_2, "orange", marker="o", markersize=5)
ax_b_left.plot(gamma_value4, Pedge_08_2, "gray", marker="o",   markersize=5)
ax_b_left.set_xlim(-0.1, 1)
ax_b_left.set_xticks([0, 1])
ax_b_left.set_yticks([0, 0.06])
ax_b_left.set_ylabel(r'$P_1$', size=15, weight='bold')
ax_b_left.tick_params(labelsize=15)
ax_b_right = fig.add_subplot(gs_b[1], sharey=ax_b_left)
ax_b_right.plot(gamma_value5, Pedge_06_3, "green", marker="o",  markersize=5, label=r'$t_1=0.6$')
ax_b_right.plot(gamma_value5, Pedge_07_3, "orange", marker="o", markersize=5, label=r'$t_1=0.7$')
ax_b_right.plot(gamma_value5, Pedge_08_3, "gray", marker="o",   markersize=5, label=r'$t_1=0.8$')
ax_b_right.set_xlim(1, 30)
ax_b_right.set_xticks([10, 20, 30])
ax_b_right.set_ylim(0, 0.06)
plt.setp(ax_b_right.get_yticklabels(), visible=False)
ax_b_right.legend(frameon=False, fontsize=12, loc='upper right')
ax_b_right.set_xlabel(r'$\gamma$', size=15, weight='bold')
ax_b_right.tick_params(labelsize=15)

ax_c = fig.add_subplot(gs_main[1, 0])
x_values = np.linspace(-1, 1, 40)
ax_c.scatter(np.real(t22), np.imag(t22),   s=3, color='red')
ax_c.scatter(np.real(t222), np.imag(t222), s=3, color='red')
ax_c.scatter(np.real(t33), np.imag(t33),   s=3, color='blue')
ax_c.scatter(np.real(t333), np.imag(t333), s=3, color='blue')
ax_c.scatter(np.real(t44), np.imag(t44),   s=3, color='black')
ax_c.scatter(np.real(t444), np.imag(t444), s=3, color='black')
ax_c.plot(x_values, np.zeros_like(x_values), color='black', linestyle='-', linewidth=2)
ax_c.set_xlabel('Re(E)', fontsize=15)
ax_c.set_ylabel('Im(E)', fontsize=15)
ax_c.set_xlim(-1, 1)
ax_c.set_xticks([-1, 0, 1])
ax_c.set_yticks([-0.5, 0])
ax_c.tick_params(labelsize=15)

ax_d = fig.add_subplot(gs_main[1, 1])
x_values = np.linspace(-1.5, 1.5, 40)
ax_d.scatter(np.real(t66), np.imag(t66),   s=3, color='green')
ax_d.scatter(np.real(t666), np.imag(t666), s=3, color='green')
ax_d.scatter(np.real(t77), np.imag(t77),   s=3, color='orange')
ax_d.scatter(np.real(t777), np.imag(t777), s=3, color='orange')
ax_d.scatter(np.real(t88), np.imag(t88),   s=3, color='gray')
ax_d.scatter(np.real(t888), np.imag(t888), s=3, color='gray')
ax_d.plot(x_values, np.zeros_like(x_values), color='black', linestyle='-', linewidth=2)
ax_d.set_xlabel('Re(E)', fontsize=15)
ax_d.set_ylabel('Im(E)', fontsize=15)
ax_d.set_xlim(-1.4, 1.4)
ax_d.set_xticks([-1, 0, 1])
ax_d.set_yticks([-0.5, 0])
ax_d.tick_params(labelsize=15)
plt.tight_layout()
#====================Save data locally==========#
np.savez('fig1.npz',
         gamma_value1=gamma_value1,
         gamma_value2=gamma_value2,
         gamma_value3=gamma_value3,
         gamma_value4=gamma_value4,
         gamma_value5=gamma_value5,
         Pedge_02_1=Pedge_02_1,
         Pedge_02_2=Pedge_02_2,
         Pedge_03_1=Pedge_03_1,
         Pedge_03_2=Pedge_03_2,
         Pedge_04_1=Pedge_04_1,
         Pedge_04_2=Pedge_04_2,
         Pedge_06_1=Pedge_06_1,
         Pedge_06_2=Pedge_06_2,
         Pedge_06_3=Pedge_06_3,
         Pedge_07_1=Pedge_07_1,
         Pedge_07_2=Pedge_07_2,
         Pedge_07_3=Pedge_07_3,
         Pedge_08_1=Pedge_08_1,
         Pedge_08_2=Pedge_08_2,
         Pedge_08_3=Pedge_08_3,
         k_values=k_values,
         t22=t22, t222=t222,
         t33=t33, t333=t333,
         t44=t44, t444=t444,
         t66=t66, t666=t666,
         t77=t77, t777=t777,
         t88=t88, t888=t888)
plt.show()
