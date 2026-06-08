import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#======================Fig1 data reading+drawing images=====================#
data = np.load(r'C:\Users\26365\Desktop\figure\fig1\fig1.npz')
gamma_value1 = data['gamma_value1']
gamma_value2 = data['gamma_value2']
gamma_value3 = data['gamma_value3']
gamma_value4 = data['gamma_value4']
gamma_value5 = data['gamma_value5']
Pedge_02_1 = data['Pedge_02_1']
Pedge_02_2 = data['Pedge_02_2']
Pedge_03_1 = data['Pedge_03_1']
Pedge_03_2 = data['Pedge_03_2']
Pedge_04_1 = data['Pedge_04_1']
Pedge_04_2 = data['Pedge_04_2']
Pedge_06_1 = data['Pedge_06_1']
Pedge_06_2 = data['Pedge_06_2']
Pedge_06_3 = data['Pedge_06_3']
Pedge_07_1 = data['Pedge_07_1']
Pedge_07_2 = data['Pedge_07_2']
Pedge_07_3 = data['Pedge_07_3']
Pedge_08_1 = data['Pedge_08_1']
Pedge_08_2 = data['Pedge_08_2']
Pedge_08_3 = data['Pedge_08_3']
k_values = data['k_values']
t22, t222 = data['t22'], data['t222']
t33, t333 = data['t33'], data['t333']
t44, t444 = data['t44'], data['t444']
t66, t666 = data['t66'], data['t666']
t77, t777 = data['t77'], data['t777']
t88, t888 = data['t88'], data['t888']
#==================drawing==================#
fig = plt.figure(figsize=(9, 9))
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
ax_a.set_yticks([0, 0.06, 0.12])
ax_a.legend(frameon=False, fontsize=11, loc='upper right')
ax_a.set_xlabel(r'$\gamma$', size=15, weight='bold')
ax_a.set_ylabel(r'$P_1$', size=15, weight='bold')
ax_a.tick_params(labelsize=15)

gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 1], width_ratios=[2, 7], wspace=0.03)
ax_b_left = fig.add_subplot(gs_b[0])
ax_b_left.plot(gamma_value3, Pedge_06_1, "green", marker="o", markersize=5)
ax_b_left.plot(gamma_value3, Pedge_07_1, "orange", marker="o", markersize=5)
ax_b_left.plot(gamma_value3, Pedge_08_1, "gray", marker="o", markersize=5)
ax_b_left.plot(gamma_value4, Pedge_06_2, "green", marker="o", markersize=5)
ax_b_left.plot(gamma_value4, Pedge_07_2, "orange", marker="o", markersize=5)
ax_b_left.plot(gamma_value4, Pedge_08_2, "gray", marker="o", markersize=5)
ax_b_left.set_xlim(-0.1, 1)
ax_b_left.set_xticks([0, 1])
ax_b_left.set_yticks([0, 0.06])
ax_b_left.set_ylabel(r'$P_1$', size=15, weight='bold')
ax_b_left.tick_params(labelsize=15)
ax_b_right = fig.add_subplot(gs_b[1], sharey=ax_b_left)
ax_b_right.plot(gamma_value5, Pedge_06_3, "green", marker="o", markersize=5, label=r'$t_1=0.6$')
ax_b_right.plot(gamma_value5, Pedge_07_3, "orange", marker="o", markersize=5, label=r'$t_1=0.7$')
ax_b_right.plot(gamma_value5, Pedge_08_3, "gray", marker="o", markersize=5, label=r'$t_1=0.8$')
ax_b_right.set_xlim(1, 30)
ax_b_right.set_xticks([10, 20, 30])
ax_b_right.set_ylim(0, 0.06)
plt.setp(ax_b_right.get_yticklabels(), visible=False)
ax_b_right.legend(frameon=False, fontsize=11, loc='upper right')
ax_b_right.tick_params(labelsize=15)
fig.text(0.79, 0.51, r'$\gamma$', fontsize=15, weight='bold')

ax_c = fig.add_subplot(gs_main[1, 0])
x_values = np.linspace(-1, 1, 40)
ax_c.scatter(np.real(t22), np.imag(t22), s=3, color='red')
ax_c.scatter(np.real(t222), np.imag(t222), s=3, color='red')
ax_c.scatter(np.real(t33), np.imag(t33), s=3, color='blue')
ax_c.scatter(np.real(t333), np.imag(t333), s=3, color='blue')
ax_c.scatter(np.real(t44), np.imag(t44), s=3, color='black')
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
ax_d.scatter(np.real(t66), np.imag(t66), s=3, color='green')
ax_d.scatter(np.real(t666), np.imag(t666), s=3, color='green')
ax_d.scatter(np.real(t77), np.imag(t77), s=3, color='orange')
ax_d.scatter(np.real(t777), np.imag(t777), s=3, color='orange')
ax_d.scatter(np.real(t88), np.imag(t88), s=3, color='gray')
ax_d.scatter(np.real(t888), np.imag(t888), s=3, color='gray')
ax_d.plot(x_values, np.zeros_like(x_values), color='black', linestyle='-', linewidth=2)
ax_d.set_xlabel('Re(E)', fontsize=15)
ax_d.set_ylabel('Im(E)', fontsize=15)
ax_d.set_xlim(-1.4, 1.4)
ax_d.set_xticks([-1, 0, 1])
ax_d.set_yticks([-0.5, 0])
ax_d.tick_params(labelsize=15)
plt.tight_layout()
plt.show()