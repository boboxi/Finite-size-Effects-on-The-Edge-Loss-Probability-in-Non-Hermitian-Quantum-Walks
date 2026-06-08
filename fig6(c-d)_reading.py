import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
#==========================fig6(c)(d) Data reading and image drawing================#
data = np.load(r'C:\Users\26365\Desktop\figure\Image related code\fig6\fig6(c-d).npz')
gamma_value1 = data['gamma_value1']
gamma_value2 = data['gamma_value2']
gamma_value3 = data['gamma_value3']
gamma_value4 = data['gamma_value4']
gamma_value5 = data['gamma_value5']
Pedge_04_1_90 = data['Pedge_04_1_90']
Pedge_04_2_90 = data['Pedge_04_2_90']
Pedge_04_1_70 = data['Pedge_04_1_70']
Pedge_04_2_70 = data['Pedge_04_2_70']
Pedge_04_1_50 = data['Pedge_04_1_50']
Pedge_04_2_50 = data['Pedge_04_2_50']
Pedge_04_1_30 = data['Pedge_04_1_30']
Pedge_04_2_30 = data['Pedge_04_2_30']
Pedge_06_1_30 = data['Pedge_06_1_30']
Pedge_06_2_30 = data['Pedge_06_2_30']
Pedge_06_3_30 = data['Pedge_06_3_30']
Pedge_06_1_50 = data['Pedge_06_1_50']
Pedge_06_2_50 = data['Pedge_06_2_50']
Pedge_06_3_50 = data['Pedge_06_3_50']
Pedge_06_1_70 = data['Pedge_06_1_70']
Pedge_06_2_70 = data['Pedge_06_2_70']
Pedge_06_3_70 = data['Pedge_06_3_70']
Pedge_06_1_90 = data['Pedge_06_1_90']
Pedge_06_2_90 = data['Pedge_06_2_90']
Pedge_06_3_90 = data['Pedge_06_3_90']
fig = plt.figure(figsize=(12, 6))
gs_main = gridspec.GridSpec(1, 2, figure=fig)
ax_a = fig.add_subplot(gs_main[0, 0])
ax_a.plot(gamma_value1, Pedge_04_1_90, marker="o", markersize=4.5, label=r'$x_0=90$', color='k')
ax_a.plot(gamma_value2, Pedge_04_2_90, marker="o", markersize=4.5, color='k')
ax_a.plot(gamma_value1, Pedge_04_1_70, marker="x", markersize=7, label=r'$x_0=70$', color='purple')
ax_a.plot(gamma_value2, Pedge_04_2_70, marker="x", markersize=7, color='purple')
ax_a.plot(gamma_value1, Pedge_04_1_50, marker="+", markersize=7, label=r'$x_0=50$', color='m')
ax_a.plot(gamma_value2, Pedge_04_2_50, marker="+", markersize=7, color='m')
ax_a.plot(gamma_value1, Pedge_04_1_30, marker="s", markersize=5, label=r'$x_0=30$', color='brown')
ax_a.plot(gamma_value2, Pedge_04_2_30, marker="s", markersize=5, color='brown')
ax_a.set_xlim(-0.2, 10)
ax_a.set_xticks([0, 2, 4, 6, 8, 10])
ax_a.set_yticks([0, 0.1, 0.2])
ax_a.legend(frameon=False, fontsize=18, loc='upper right')
ax_a.set_xlabel(r'$\gamma$', size=22)
ax_a.set_ylabel(r'$P_1$', size=22)
ax_a.tick_params(labelsize=20)

gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 1], width_ratios=[2, 7], wspace=0.03)
ax_b_left = fig.add_subplot(gs_b[0])
ax_b_left.plot(gamma_value3, Pedge_06_1_90, marker="o", markersize=4.5, color='k')
ax_b_left.plot(gamma_value4, Pedge_06_2_90, marker="o", markersize=4.5, color='k')
ax_b_left.plot(gamma_value3, Pedge_06_1_70, marker="x", markersize=7, color='purple')
ax_b_left.plot(gamma_value4, Pedge_06_2_70, marker="x", markersize=7, color='purple')
ax_b_left.plot(gamma_value3, Pedge_06_1_50, marker="+", markersize=7, color='m')
ax_b_left.plot(gamma_value4, Pedge_06_2_50, marker="+", markersize=7, color='m')
ax_b_left.plot(gamma_value3, Pedge_06_1_30, marker="s", markersize=5, color='brown')
ax_b_left.plot(gamma_value4, Pedge_06_2_30, marker="s", markersize=5, color='brown')
ax_b_left.set_xlim(-0.1, 1)
ax_b_left.set_xticks([0, 1])
ax_b_left.set_yticks([0, 0.05, 0.1])
ax_b_left.tick_params(labelsize=20)
ax_b_right = fig.add_subplot(gs_b[1], sharey=ax_b_left)
ax_b_right.plot(gamma_value5, Pedge_06_3_90, marker="o", markersize=4.5, color='k', label=r'$x_0=90$')
ax_b_right.plot(gamma_value5, Pedge_06_3_70, marker="x", markersize=7, color='purple', label=r'$x_0=70$')
ax_b_right.plot(gamma_value5, Pedge_06_3_50, marker="+", markersize=7, color='m', label=r'$x_0=50$')
ax_b_right.plot(gamma_value5, Pedge_06_3_30, marker="s", markersize=5, color='brown', label=r'$x_0=30$')
ax_b_right.tick_params(left=False, labelleft=False)
ax_b_right.set_xlim(1, 30)
ax_b_right.set_xticks([10, 20, 30])
plt.setp(ax_b_right.get_yticklabels(), visible=False)
ax_b_right.legend(frameon=False, fontsize=18, loc='lower right')
fig.text(0.77, 0.04, r'$\gamma$', fontsize=22)
ax_b_right.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('fig6(c-d).png', dpi=300, bbox_inches='tight')
#plt.show()

