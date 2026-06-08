import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.ticker import FixedLocator as MinorFixedLocator
#+===============fig5 Drawing code=========================#
t1 = 0.7
t2 = 0.5
L = 100
dt = 0.1
total_time = 800
x0 = 90
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
xx0_values = np.arange(20, 51, 1)
#================Extreme dissipation fig5(a)(d)================#
gamma = 0.02
Pbulk_1 = []
H = hamiltonian(t1, t2, gamma, L)
psi0 = initial_state(L, x0)
psi_t = time_evolution(H, psi0, dt, total_time)
for xx0 in xx0_values:
    P = loss_probability(psi_t, gamma, L, dt)
    P_bulk = P[abs(xx0 - x0) - 1]/P[69]
    Pbulk_1.append(P_bulk)
gamma = 1.4
Pbulk_2 = []
H = hamiltonian(t1, t2, gamma, L)
psi0 = initial_state(L, x0)
psi_t = time_evolution(H, psi0, dt, total_time)
for xx0 in xx0_values:
    P = loss_probability(psi_t, gamma, L, dt)
    P_bulk = P[abs(xx0 - x0) - 1]/P[69]
    Pbulk_2.append(P_bulk)
 # ================Extreme dissipation fig5(c)(f)================#
gamma = 30
Pbulk_3 = []
H = hamiltonian(t1, t2, gamma, L)
psi0 = initial_state(L, x0)
psi_t = time_evolution(H, psi0, dt, total_time)
for xx0 in xx0_values:
    P = loss_probability(psi_t, gamma, L, dt)
    P_bulk = P[abs(xx0 - x0) - 1]/P[69]
    Pbulk_3.append(P_bulk)
fig, axs = plt.subplots(2, 3,figsize=(18,11))
axs[0,0].plot(xx0_values, Pbulk_1, 'orange',marker='o', markersize=6, label=r'$\gamma=0.02$')
#==================Double logarithmic coordinate =========================#
axs[0,0].set_xscale("log")
axs[0,0].set_yscale("log")
axs[0,0].set_xlabel(r'$|x-x_0|$',size=20)
axs[0,0].set_ylabel(r'$P_{x}/P_{70}$',size=20)
axs[0,0].set_xlim(19.7,50)
axs[0,0].set_xticks([20,30,40,50])
axs[0,0].set_xticklabels(['20', '30', '40', '50'])
axs[0,0].tick_params(labelsize=20)
axs[0,0].legend(frameon=False,fontsize=20)
axs[0,0].set_ylim(0.39, 1.05)
axs[0,0].set_yticks([1.0])
axs[0,0].set_yticklabels([r'$10^{0}$'])
axs[0,0].yaxis.set_major_locator(FixedLocator([1.0]))
axs[0,0].yaxis.set_minor_locator(MinorFixedLocator([0.4,0.5,0.6,0.7,0.8,0.9]))
axs[0,0].set_yticklabels([0.4,0.5,0.6,0.7,0.8,0.9], minor=True)
for tick in axs[0,0].yaxis.get_minor_ticks():
    label_text = tick.label1.get_text()
    if label_text not in ['0.4', '0.7']:
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
axs[0,0].tick_params(axis='y', which='major', length=8, width=1.8 )
axs[0,0].tick_params(axis='y', which='minor', length=4, width=1.2,labelsize=16)
axs[0,0].text(-0.1, 1.03, '(a)', transform=axs[0,0].transAxes, fontsize=20)
axs[1,0].plot(xx0_values, Pbulk_1, 'orange',marker='^', markersize=7.5, label=r'$\gamma=0.02$')
#==================Semi-logarithmic coordinate =========================#
axs[1,0].set_yscale("log")
axs[1,0].set_xlabel(r'$|x-x_0|$',size=20)
axs[1,0].set_ylabel(r'$P_{x}/P_{70}$',size=20)
axs[1,0].set_xlim(19.7,50)
axs[1,0].set_xticks([20,30,40,50])
axs[1,0].tick_params(labelsize=20)
axs[1,0].legend(frameon=False,fontsize=20)
axs[1,0].set_ylim(0.39, 1.05)
axs[1,0].set_yticks([1.0])
axs[1,0].set_yticklabels([r'$10^{0}$'])
axs[1,0].yaxis.set_major_locator(FixedLocator([1.0]))
axs[1,0].yaxis.set_minor_locator(MinorFixedLocator([0.4,0.5,0.6,0.7,0.8,0.9]))
axs[1,0].set_yticklabels([0.4,0.5,0.6,0.7,0.8,0.9], minor=True)
for tick in axs[1,0].yaxis.get_minor_ticks():
    label_text = tick.label1.get_text()
    if label_text not in ['0.4', '0.7']:
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
axs[1,0].tick_params(axis='y', which='major', length=8, width=1.8 )
axs[1,0].tick_params(axis='y', which='minor', length=4, width=1.2,labelsize=16)
axs[1,0].text(-0.1, 1.03, '(d)', transform=axs[1,0].transAxes, fontsize=20)
axs[0,1].plot(xx0_values, Pbulk_2, 'orange',marker='o', markersize=6, label=r'$\gamma=1.4$')
#==================Double logarithmic coordinate=========================#
axs[0,1].set_xscale("log")
axs[0,1].set_yscale("log")
axs[0,1].set_xlabel(r'$|x-x_0|$',size=20)
axs[0,1].set_xlim(19.7,50)
axs[0,1].set_xticks([20,30,40,50])
axs[0,1].set_xticklabels(['20', '30', '40', '50'])
axs[0,1].tick_params(labelsize=20)
axs[0,1].tick_params(axis='y', which='major', length=8, width=1.8 )
axs[0,1].tick_params(axis='y', which='minor', length=4, width=1.2)
axs[0,1].legend(frameon=False,fontsize=20)
axs[0,1].text(-0.1, 1.03, '(b)', transform=axs[0,1].transAxes, fontsize=20)
axs[1,1].plot(xx0_values, Pbulk_2, 'orange',marker='^', markersize=7.5, label=r'$\gamma=1.4$')
#==================Semi-logarithmic coordinate =========================#
axs[1,1].set_yscale("log")
axs[1,1].set_xlabel(r'$|x-x_0|$',size=20)
axs[1,1].set_xlim(19.7,50)
axs[1,1].set_xticks([20,30,40,50])
axs[1,1].tick_params(labelsize=20)
axs[1,1].legend(frameon=False,fontsize=20)
axs[1,1].tick_params(axis='y', which='major', length=8, width=1.8)
axs[1,1].tick_params(axis='y', which='minor', length=4, width=1.2)
axs[1,1].text(-0.1, 1.03, '(e)', transform=axs[1,1].transAxes, fontsize=20)
axs[0,2].plot(xx0_values, Pbulk_3,'orange',marker='o', markersize=6, label=r'$\gamma=30$')
#==================Double logarithmic coordinate=========================#
axs[0,2].set_xscale("log")
axs[0,2].set_yscale("log")
axs[0,2].set_xlabel(r'$|x-x_0|$',size=20)
axs[0,2].set_xlim(19.7,50)
axs[0,2].set_xticks([20,30,40,50])
axs[0,2].set_xticklabels(['20', '30', '40', '50'])
axs[0,2].tick_params(labelsize=20)
axs[0,2].legend(frameon=False,fontsize=20)
axs[0,2].set_ylim(0.39, 1.05)
axs[0,2].set_yticks([1.0])
axs[0,2].set_yticklabels([r'$10^{0}$'])
axs[0,2].yaxis.set_major_locator(FixedLocator([1.0]))
axs[0,2].yaxis.set_minor_locator(MinorFixedLocator([0.4,0.5,0.6,0.7,0.8,0.9]))
axs[0,2].set_yticklabels([0.4,0.5,0.6,0.7,0.8,0.9], minor=True)
for tick in axs[0,2].yaxis.get_minor_ticks():
    label_text = tick.label1.get_text()
    if label_text not in ['0.4', '0.7']:
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
axs[0,2].tick_params(axis='y', which='major', length=8, width=1.8 )
axs[0,2].tick_params(axis='y', which='minor', length=4, width=1.2,labelsize=16)
axs[0,2].text(-0.1, 1.03, '(c)', transform=axs[0,2].transAxes, fontsize=20)
axs[1,2].plot(xx0_values, Pbulk_3,'orange',marker='^', markersize=7.5, label=r'$\gamma=30$')
#==================Semi-logarithmic coordinate =========================#
axs[1,2].set_yscale("log")
axs[1,2].set_xlabel(r'$|x-x_0|$',size=20)
axs[1,2].set_xlim(19.7,50)
axs[1,2].set_xticks([20,30,40,50])
axs[1,2].tick_params(labelsize=20)
axs[1,2].legend(frameon=False,fontsize=20)
axs[1,2].set_ylim(0.39, 1.05)
axs[1,2].set_yticks([1.0])
axs[1,2].set_yticklabels([r'$10^{0}$'])
axs[1,2].yaxis.set_major_locator(FixedLocator([1.0]))
axs[1,2].yaxis.set_minor_locator(MinorFixedLocator([0.4,0.5,0.6,0.7,0.8,0.9]))
axs[1,2].set_yticklabels([0.4,0.5,0.6,0.7,0.8,0.9], minor=True)
for tick in axs[1,2].yaxis.get_minor_ticks():
    label_text = tick.label1.get_text()
    if label_text not in ['0.4', '0.7']:
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
axs[1,2].tick_params(axis='y', which='major', length=8, width=1.8 )
axs[1,2].tick_params(axis='y', which='minor', length=4, width=1.2,labelsize=16)
axs[1,2].text(-0.1, 1.03, '(f)', transform=axs[1,2].transAxes, fontsize=20)
plt.show()
#plt.savefig('fig5.png', dpi=400, bbox_inches='tight')
