import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.gridspec as gridspec
#=======================fig4  calculation code===============#
x0=90
L=100
t2 = 0.5
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
#================Numerical results of P1 on the left side of the Fig4a broken axis diagram=========================#
gamma_value = np.arange(0,0.11,0.01)
gamma_value2 = np.arange(0.1,1.1,0.1)
t1=0.6
Pedge_06_1 = []
for gamma in gamma_value:
    H = hamiltonian(t1, t2, L,gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge06_1 = P[0]
    Pedge_06_1.append(Pedge06_1)
Pedge_06_2 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L,gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge06_2 = P[0]
    Pedge_06_2.append(Pedge06_2)
t1=0.7
Pedge_07_1 = []
for gamma in gamma_value:
    H = hamiltonian(t1, t2, L,gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge07_1 = P[0]
    Pedge_07_1.append(Pedge07_1)
Pedge_07_2 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L,gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge07_2 = P[0]
    Pedge_07_2.append(Pedge07_2)
t1=0.8
Pedge_08_1 = []
for gamma in gamma_value:
    H = hamiltonian(t1, t2, L,gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge08_1 = P[0]
    Pedge_08_1.append(Pedge08_1)
Pedge_08_2 = []
for gamma in gamma_value2:
    H = hamiltonian(t1, t2, L,gamma)
    psi0 = initial_state(L, x0)
    psi_t = evolution(H, psi0, dt, total_time)
    P = loss_probability(psi_t, gamma, L, dt)
    Pedge08_2 = P[0]
    Pedge_08_2.append(Pedge08_2)
fig = plt.figure(figsize=(11,9))
gs_outer = gridspec.GridSpec(2, 2, figure=fig)
gs_left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0, 0], width_ratios=[1.8, 5], wspace=0.02)
ax1_left = fig.add_subplot(gs_left[0])
ax1_left.set_ylabel(r'$P_1$', size=16)
ax1_left.plot(gamma_value, Pedge_06_1, "green",  marker="o", markersize=3)
ax1_left.plot(gamma_value, Pedge_07_1, "orange", marker="o", markersize=3)
ax1_left.plot(gamma_value, Pedge_08_1, "gray",   marker="o", markersize=3)
ax1_left.plot(gamma_value2, Pedge_06_2, "green", marker="o", markersize=3)
ax1_left.plot(gamma_value2, Pedge_07_2, "orange",marker="o", markersize=3)
ax1_left.plot(gamma_value2, Pedge_08_2, "gray",  marker="o", markersize=3)
#=============P1 Analytical results of the left in the Fig4a broken axis diagram ,data from t1=0.6. nb, t1=0.7. nb, t1=0.8. nb(Mathematica)==============#
data6=[0.3040187,0.1790457,0.1140491,0.07555491,0.05129573,0.03543838,0.02481747,0.01757681,0.01257226,0.009074069]
data7=[0.2306273,0.1096382,0.05635154,0.03012107,0.016495,0.009187808,0.005184586,0.002956794,0.001701697,0.0009873697]
data8=[0.1868082,0.07569146,0.03312529,0.01507751,0.007030358,0.003333634,0.001600973,0.0007768043,0.0003802123,0.000187537]
ax1_left.plot(np.arange(0.01,0.11,0.01),data6,'green',label='Ana', linestyle='--')
ax1_left.plot(np.arange(0.01,0.11,0.01),data7,'orange',label='Ana', linestyle='--')
ax1_left.plot(np.arange(0.01,0.11,0.01),data8,'gray',label='Ana', linestyle='--')
data66=[0.009070025,0.0005495144,0.00008935203,0.00005823274,0.0001079062,0.0002391163,0.0004796502,0.0008520102,0.001368009,0.002029585]
data77=[0.000985529,0,0,0,0,0,0,0,0,0]
data88=[0.0001864920881,0,0,0,0,0,0,0,0,0]
ax1_left.plot(np.arange(0.1,1.1,0.1),data66,'green',label='Ana',linestyle='--')
ax1_left.plot(np.arange(0.1,1.1,0.1),data77,'orange',label='Ana',linestyle='--')
ax1_left.plot(np.arange(0.1,1.1,0.1),data88,'gray',label='Ana',linestyle='--')
ax1_left.set_ylim(0, 0.05)
ax1_left.set_yticks([0,0.02, 0.04])
ax1_left.set_xlim(-0.1, 1)
ax1_left.set_xticks([0, 1])
ax1_left.tick_params(labelsize=15)
#========P1 results of numerical, analysis and zero frequency expansion calculation of fig. 4a broken-axis diagram(right) . Data is stored in text file.======#
# ==================================The calculation process comes from t1=0.6.nb, t1=0.7.nb, t1=0.8.nb (Mathematica)============#
path = "C:/Users/26365/Desktop/figure/fig4/"
data = np.loadtxt(path + 'data-fig4a.txt')
x6 = data[0:26, 0]
y6 = data[0:26, 1]
x66 = data[26:2*26, 0]
y66 = data[26:2*26, 1]
x666 = data[26*2:26*3,0]
y666 = data[26*2:26*3,1]
x7 = data[26*3:110, 0]
y7 = data[26*3:110, 1]
x77 = data[110:142, 0]
y77= data[110:142, 1]
x777 = data[142:174,0]
y777 = data[142:174,1]
x8 = data[174:215, 0]
y8 = data[174:215, 1]
x88 = data[215:256, 0]
y88= data[215:256, 1]
x888 = data[256:297,0]
y888 = data[256:297,1]
ax1_right = fig.add_subplot(gs_left[1],sharey=ax1_left)
plt.setp(ax1_right.get_yticklabels(), visible=False)
ax1_right.plot(x6, y6,   'green', marker='o', markersize=3, label='Num.')
ax1_right.plot(x66, y66, 'green', label='Ana.',linestyle='--')
ax1_right.plot(x666,y666,'green', marker='s',  markersize=4,fillstyle='none',label=r'ED in $\omega=0$')
ax1_right.plot(x7, y7,   'orange', marker='o', markersize=3, label='Num.')
ax1_right.plot(x77, y77, 'orange',label='Ana.',linestyle='--')
ax1_right.plot(x777,y777,'orange',marker='s', markersize=4,fillstyle='none',label='_nolegend_')
ax1_right.plot(x8, y8,   'gray', marker='o', markersize=3, label='Num.')
ax1_right.plot(x88, y88, 'gray', label='Ana.',linestyle='--')
ax1_right.plot(x888,y888,'gray',  marker='s', markersize=4,fillstyle='none',label='_nolegend_')
fig.text(0.29, 0.52, r'$\gamma$', ha='center', fontsize=16)
ax1_right.set_ylim(0, 0.05)
ax1_right.set_yticks([0, 0.02,0.04])
ax1_right.set_xlim(1, 20)
ax1_right.set_xticks([10,20])
ax1_right.tick_params(axis='y', left=False, labelleft=False)
ax1_right.tick_params(labelsize=15)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', marker='o', markersize=6, label=r'$t_1=0.6$' ,linestyle='none'),
    Line2D([0], [0], color='orange', marker='o',markersize=6, label=r'$t_1=0.7$',linestyle='none'),
    Line2D([0], [0], color='gray', marker='o',  markersize=6, label=r'$t_1=0.8$',linestyle='none'),
    Line2D([0], [0], color='black', marker='o', markersize=6, label='Num.'),
    Line2D([0], [0], color='black', label='Ana.', linestyle='--'),
    Line2D([0], [0], color='black', marker='s', markersize=6, markerfacecolor='none', label=r'ED in $\omega=0$')]
ax1_right.legend(handles=legend_elements, loc='upper center', frameon=False, fontsize=12, ncol=2)
ax1_left.text(-0.2, 1.05, '(a)', transform=ax1_left.transAxes, fontsize=16)

#================Scattering probability versus dissipation curve================#
ax2 = fig.add_subplot(gs_outer[0, 1])
path = "C:/Users/26365/Desktop/figure/fig4/"
data = np.loadtxt(path + 'data-fig4b.txt')
x1 = data[0:100, 0]
y1 = data[0:100, 1]
x2 = data[100:2*100, 0]
y2 = data[100:2*100, 1]
x3 = data[2*100:100*3, 0]
y3 = data[100*2:100*3, 1]
ax2.plot(x1, abs(y1), 'o-', color='green', label=r'$t_1=0.6$', markersize=3, linewidth=1)
ax2.plot(x2, abs(y2), 'o-', color='orange', label=r'$t_1=0.7$', markersize=3, linewidth=1)
ax2.plot(x3, abs(y3), 'o-', color='gray', label=r'$t_1=0.8$', markersize=3, linewidth=1)
ax2.set_xlabel(r'$\gamma$',size=16)
ax2.set_ylabel(r'$|\frac{d\lambda(v)}{dv}|_{v=0}|$',size=16)
ax2.set_xlim(0,2)
ax2.set_xticks([0,0.4,0.8,1.2,1.6,2])
ax2.set_ylim(0,3)
ax2.set_yticks([0,1,2,3])
ax2.legend(frameon=False,fontsize=15,loc='upper left')
ax2.tick_params(labelsize=15)
ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=16)

#====================The variation of min|β(w,γ)| with dissipation when IGO========================================#
ax3 = fig.add_subplot(gs_outer[1, 0])
def b(w, t1, t2, r):
    return t1**2 + t2**2 - w**2 - 1j * r * w
def beta_f(w, t1, t2, r):
    return (-b(w, t1, t2, r)-np.sqrt(b(w, t1, t2, r)**2-4*t2**2*(t1**2 - r**2 / 4)))/(2*t2 * (t1 + r/ 2))
def beta_z(w, t1, t2, r):
    return (-b(w, t1, t2, r)+np.sqrt(b(w, t1, t2, r)**2-4*t2**2*(t1**2 - r**2 / 4)))/(2*t2 * (t1 + r/ 2))
r_values = np.arange(0, 40.1, 0.1)
t1_values = [0.6, 0.7, 0.8]
t2 = 0.5
beta_l = []
for t1 in t1_values:
    values1 = []
    for r in r_values:
        w = np.linspace(-2, 2, 400)
        value_f = np.abs(beta_f(w, t1, t2, r))
        value_z = np.abs(beta_z(w, t1, t2, r))
        all_values = np.concatenate([value_f, value_z])
        selected_values = all_values[all_values >= 1]
        if len(selected_values) > 0:
            current_max = np.min(selected_values)
            values1.append(current_max)
        else:
            values1.append(np.nan)
    beta_l.append(values1)
colors = ['green', 'orange', 'grey']
labels = [r'$t_1=0.6$', r'$t_1=0.7$', r'$t_1=0.8$']
for i in range(len(t1_values)):
    ax3.plot(r_values, beta_l[i], color=colors[i], label=labels[i],lw=2)
ax3.set_xlabel(r'$\gamma$', size=16)
ax3.set_ylabel(r'min{$|\beta_L|$}', size=16)
ax3.set_xlim(-0.3, 30)
ax3.set_xticks([0, 10, 20, 30])
ax3.set_ylim(1, 1.21)
ax3.set_yticks([1, 1.1, 1.2])
ax3.tick_params( labelsize=15)
ax3.legend(frameon=False, fontsize=15)
ax3.text(-0.05, 1.05, '(c)', transform=ax3.transAxes, fontsize=16)

#============reemerge of edge burst and boundary scattering when IGO===========#
ax4 = fig.add_subplot(gs_outer[1,1])
t1 = 0.6
t2 = 0.5
gamma = 10
H = hamiltonian(t1, t2, L,gamma)
psi0 = initial_state(L, x0)
psi_t = evolution(H, psi0, dt, total_time)
P = loss_probability(psi_t, gamma, L, dt)
ax4.bar(np.arange(L) + 1, P)
ax4.set_xlim(0,100)
ax4.set_xlabel(r'$x$', size=16)
ax4.set_xticks([0, 20, 40, 60, 80, 100])
ax4.set_yticks([0,0.08, 0.16])
ax4.set_ylabel(r'$P_x$', size=16)
ax4.text(2, 0.16, r'$t_1=0.6, \gamma=10$', size=15)
ax4.tick_params(labelsize=15)
psi_t = time_evolution(H, psi0, dt, total_time)
psi_t_squared = np.abs(psi_t) ** 2
psi2 = psi_t_squared.T
inset2 = inset_axes(ax4, width="50%", height="50%", loc='center')
x_coords = np.arange(2*L)
t_coords = np.arange(0, total_time, dt)
X2, T2 = np.meshgrid(x_coords, t_coords)
im2 = inset2.pcolormesh(X2, T2, psi2, cmap='binary', shading="auto", vmin=0, vmax=0.06)
inset2.text(0.6, 0.85, r'$|\psi(x,t)|^2$', transform=inset2.transAxes,fontsize=11, color='black')
inset2.set_xlabel(r'$x$', size=12)
inset2.set_xticks([0, 100,200])
inset2.set_ylabel(r'$t$', size=12)
inset2.tick_params(labelsize=15)
inset2.set_yticks([0, 800])
ax4.text(-0.1, 1.05, '(d)', transform=ax4.transAxes, fontsize=16)
plt.tight_layout()
plt.show()
