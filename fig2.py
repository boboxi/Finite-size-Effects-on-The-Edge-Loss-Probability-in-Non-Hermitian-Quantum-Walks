import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#====================Fig2 calculation code================#
t2 = 0.5
x0 = 90
L  = 100
dt = 0.1
total_time = 800
def hamiltonian(t1, t2, L,gamma):
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
    psi[2 * x0-2] = 1
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
def time_evolution(H, psi0, dt, total_time):
    def schrodinger(t, psi):
        return -1j * H @ psi
    def normalize(psi):
        norm = np.linalg.norm(psi)
        return psi / norm if norm != 0 else psi
    t_span = [0, total_time]
    t_eval = np.arange(0, total_time, dt)
    sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval, method='RK45')
    normalized_psi_t = np.zeros_like(sol.y)
    for i in range(sol.y.shape[1]):
        normalized_psi_t[:, i] = normalize(sol.y[:, i])
    return normalized_psi_t
# ==================== Numerical Calculation Pedge ====================
def compute_Pedge(t1, gamma_values):
    Pedge_list = []
    for gamma in gamma_values:
        H = hamiltonian(t1, t2, L, gamma)
        psi0 = initial_state(L, x0)
        psi_t = evolution(H, psi0, dt, total_time)
        P = loss_probability(psi_t, gamma, L, dt)
        Pedge_list.append(P[0])
    return np.array(Pedge_list)
gamma_values = np.arange(0, 1.42, 0.02)
print("计算 t1=0.2 ...")
P102N = compute_Pedge(0.2, gamma_values)
print("计算 t1=0.3 ...")
P103N = compute_Pedge(0.3, gamma_values)
print("计算 t1=0.4 ...")
P104N = compute_Pedge(0.4, gamma_values)
#=================================Analyzing the edge escape probability ==============================================#
def Q(gamma, t1):
    sin2 = 1 - (t1**2 / t2**2)
    cos2 = t1**2 / t2**2
    return sin2 / (t1**2 * (gamma**2 * cos2 + 4 * t1**2 * sin2))
def K(gamma, t1):
    w02 = t2**2 - t1**2
    return (w02 / t1**3) * gamma / (4 * w02 + gamma**2)
def P1(gamma, t1):
    return Q(gamma, t1) * (4 * gamma / np.pi) * (2 * K(gamma, t1))**-1.5 * math.gamma(1.5) * x0**-0.5
fig = plt.figure(figsize=(10, 9))
gs_outer = gridspec.GridSpec(2, 2, figure=fig)
gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0, 0], height_ratios=[1, 3], hspace=0.1)
ax1_top = fig.add_subplot(gs_left[0])
ax1_low = fig.add_subplot(gs_left[1], sharex=ax1_top)
gammavalue=np.arange(0.001,1.41,0.01)
P102a = np.array([P1(gamma, 0.2) for gamma in gammavalue])
P103a = np.array([P1(gamma, 0.3) for gamma in gammavalue])
P104a = np.array([P1(gamma, 0.4) for gamma in gammavalue])
ax1_top.plot(gammavalue, P102a, "red",  linestyle="--", label=r'$t_1=0.2$')
ax1_top.plot(gammavalue, P103a, "blue", linestyle="--", label=r'$t_1=0.3$')
ax1_top.plot(gammavalue, P104a, "black",linestyle="--", label=r'$t_1=0.4$')
ax1_top.set_yticks([0, 1])
ax1_top.tick_params(labelsize=15)
ax1_low.plot(gammavalue, P102a, "red",  linestyle="--", label=r'$t_1=0.2$')
ax1_low.plot(gammavalue, P103a, "blue", linestyle="--", label=r'$t_1=0.3$')
ax1_low.plot(gammavalue, P104a, "black",linestyle="--", label=r'$t_1=0.4$')
gammax = np.arange(0, 1.42, 0.02)
line1 = ax1_low.plot(gammax, P102N, 'ro-', markersize=3, linewidth=1.5, label=r'$t_1=0.2$')[0]
line2 = ax1_low.plot(gammax, P103N, 'bo-', markersize=3, linewidth=1.5, label=r'$t_1=0.3$')[0]
line3 = ax1_low.plot(gammax, P104N, 'ko-', markersize=3, linewidth=1.5, label=r'$t_1=0.4$')[0]
ax1_low.set_ylim(0, 0.2)
ax1_low.set_yticks([0, 0.1, 0.2])
ax1_low.set_xlim(0, 1.4)
ax1_low.set_xlabel(r'$\gamma$', size=20)
ax1_low.set_xticks([0, 0.2,0.4, 0.6, 0.8,1,1.2, 1.4])
ax1_low.tick_params(labelsize=15)
plt.setp(ax1_top.get_xticklabels(), visible=False)
ax1_top.tick_params(axis='x', which='both', length=0)
handles, labels = ax1_low.get_legend_handles_labels()
t1_values = [0.2, 0.3, 0.4]
legend1 = ax1_low.legend(handles[:3], [rf'$t_1 = {t1}$' for t1 in t1_values],
                       loc='upper right', frameon=False, fontsize=10.5, title='Ana.', title_fontsize=10.5)
legend2 = ax1_low.legend(handles[3:6], [rf'$t_1 = {t1}$' for t1 in t1_values],
                       loc='upper center', frameon=False, fontsize=10.5, title='Num.',title_fontsize=10.5)
ax1_low.add_artist(legend1)
fig.text(0.03, 0.75, r'$P_1$', va='center', rotation='vertical', size=20)

#==========================Generalized Brillouin zone radius=====================#
ax2 = fig.add_subplot(gs_outer[0, 1])
def R1(r):
    return np.sqrt(np.abs((0.2 - r/2) / (0.2 + r/2)))
def R2(r):
    return np.sqrt(np.abs((0.3 - r/2) / (0.3 + r/2)))
def R3(r):
    return np.sqrt(np.abs((0.4 - r/2) / (0.4 + r/2)))
r_values = np.arange(0, 10.01, 0.01)
R1_values = R1(r_values)
R2_values = R2(r_values)
R3_values = R3(r_values)
ax2.plot(r_values, R1_values, 'red',  label=r'$t_1=0.2$')
ax2.plot(r_values, R2_values, 'blue', label=r'$t_1=0.3$')
ax2.plot(r_values, R3_values, 'black',label=r'$t_1=0.4$')
ax2.legend(frameon=False,fontsize=15)
ax2.set_ylim(0,1)
ax2.set_yticks([0,1])
ax2.set_xlim(0,10)
ax2.set_xticks([0,2,4,6,8,10])
ax2.set_xlabel(r'$\gamma$',size=20)
ax2.set_ylabel('$r_{G}$',size=20)
ax2.tick_params(labelsize=15)

#=================Probability distribution of bound interval and wavepacket motion(illustration)========#
ax3 = fig.add_subplot(gs_outer[1, 0])
t1 = 0.4
gamma = 0.8
H = hamiltonian(t1, t2, L, gamma)
psi0 = initial_state(L, x0)
psi_t = evolution(H, psi0, dt, total_time)
P = loss_probability(psi_t, gamma, L, dt)
ax3.bar(np.arange(L) + 1, P)
ax3.set_xlabel(r'$x$', size=20)
ax3.set_xlim([0,100])
ax3.set_xticks([0, 20, 40, 60, 80, 100])
ax3.set_yticks([0, 0.1, 0.2])
ax3.set_ylabel(r'$P_x$', size=20)
ax3.tick_params(labelsize=15)
ax3.text(2, 0.25, r'$t_1=0.4, \gamma=0.8$', size=15)
psi_t = time_evolution(H, psi0, dt, total_time)
psi_t_squared = np.abs(psi_t) ** 2
psi1 = psi_t_squared.T
inset1 = inset_axes(ax3, width="50%", height="50%", loc='center')
x_coords = np.arange(2*L)
t_coords = np.arange(0, total_time, dt)
X, T = np.meshgrid(x_coords, t_coords)
im1 = inset1.pcolormesh(X, T, psi1, cmap='binary', shading='auto', vmin=0, vmax=0.1)
inset1.text(0.6, 0.85, r'$|\psi(x,t)|^2$', transform=inset1.transAxes, fontsize=11, color='black')
inset1.set_xlabel(r'$x$', size=12)
inset1.set_xticks([0, 100,200])
inset1.set_ylabel(r'$t$', size=12)
inset1.set_yticks([0, 800])
inset1.tick_params(labelsize=12)

#=================Scattering interval probability distribution and wavepacket motion(illustration)========#
ax4 = fig.add_subplot(gs_outer[1, 1])
t1 = 0.4
gamma = 0.04
H = hamiltonian(t1, t2, L,gamma)
psi0 = initial_state(L, x0)
psi_t = evolution(H, psi0, dt, total_time)
P = loss_probability(psi_t, gamma, L, dt)
ax4.bar(np.arange(L) + 1, P)
ax4.set_xlim([0,100])
ax4.set_xlabel(r'$x$', size=20)
ax4.set_xticks([0, 20, 40, 60, 80, 100])
ax4.set_yticks([0, 0.05, 0.1])
ax4.set_ylabel(r'$P_x$', size=20)
ax4.text(2, 0.09, r'$t_1=0.4, \gamma=0.04$', size=15)
ax4.tick_params(labelsize=15)
psi_t = time_evolution(H, psi0, dt, total_time)
psi_t_squared = np.abs(psi_t) ** 2
psi2 = psi_t_squared.T
inset2 = inset_axes(ax4, width="50%", height="50%", loc='center')
X2, T2 = np.meshgrid(x_coords, t_coords)
im2 = inset2.pcolormesh(X2, T2, psi2, cmap='binary', shading="auto", vmin=0, vmax=0.1)
inset2.text(0.6, 0.85, r'$|\psi(x,t)|^2$', transform=inset2.transAxes, fontsize=11, color='black')
inset2.set_xlabel(r'$x$', size=12)
inset2.set_xticks([0, 100,200])
inset2.set_ylabel(r'$t$', size=12)
inset2.set_yticks([0, 800])
inset2.tick_params(labelsize=12)
plt.tight_layout()
plt.show()