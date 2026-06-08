import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
#=========================fig6 (a)(b)  Data reading and plotting====================#
data = np.load(r'C:\Users\26365\Desktop\figure\Image related code\fig6\fig6(a-b).npz')
gamma_value1 = data['gamma_value1']
gamma_value2 = data['gamma_value2']
gamma_value3 = data['gamma_value3']
gamma_value4 = data['gamma_value4']
gamma_value5 = data['gamma_value5']
Pedge_04_1_100 = data['Pedge_04_1_100']
Pedge_04_2_100 = data['Pedge_04_2_100']
Pedge_04_1_150 = data['Pedge_04_1_150']
Pedge_04_2_150 = data['Pedge_04_2_150']
Pedge_04_1_300 = data['Pedge_04_1_300']
Pedge_04_2_300 = data['Pedge_04_2_300']
Pedge_06_1_100 = data['Pedge_06_1_100']
Pedge_06_2_100 = data['Pedge_06_2_100']
Pedge_06_3_100 = data['Pedge_06_3_100']
Pedge_06_1_150 = data['Pedge_06_1_150']
Pedge_06_2_150 = data['Pedge_06_2_150']
Pedge_06_3_150 = data['Pedge_06_3_150']
Pedge_06_1_300 = data['Pedge_06_1_300']
Pedge_06_2_300 = data['Pedge_06_2_300']
Pedge_06_3_300 = data['Pedge_06_3_300']
fig = plt.figure(figsize=(12, 6))
gs_main = gridspec.GridSpec(1, 2, figure=fig)
ax_a = fig.add_subplot(gs_main[0, 0])
ax_a.plot(gamma_value1, Pedge_04_1_100, marker="o", markersize=5.5, label=r'$L=100$', color='k')
ax_a.plot(gamma_value2, Pedge_04_2_100, marker="o", markersize=5.5, color='k')
ax_a.plot(gamma_value1, Pedge_04_1_150, marker="^", markersize=8, label=r'$L=150$', color='red')
ax_a.plot(gamma_value2, Pedge_04_2_150, marker="^", markersize=8, color='red')
ax_a.plot(gamma_value1, Pedge_04_1_300, marker="s", markersize=6, label=r'$L=300$', color='blue')
ax_a.plot(gamma_value2, Pedge_04_2_300, marker="s", markersize=6, color='blue')
ax_a.set_xlim(-0.2, 10)
ax_a.set_xticks([0, 2, 4, 6, 8, 10])
ax_a.set_yticks([0, 0.06, 0.12])
ax_a.legend(frameon=False, fontsize=20, loc='lower right')
ax_a.set_xlabel(r'$\gamma$', size=22)
ax_a.set_ylabel(r'$P_1$', size=22)
ax_a.tick_params(labelsize=20)

gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 1], width_ratios=[2, 7], wspace=0.03)
ax_b_left = fig.add_subplot(gs_b[0])
ax_b_left.plot(gamma_value3, Pedge_06_1_100, marker="o", markersize=5.5, color='k')
ax_b_left.plot(gamma_value4, Pedge_06_2_100, marker="o", markersize=5.5, color='k')
ax_b_left.plot(gamma_value3, Pedge_06_1_150, marker="^", markersize=8, color='red')
ax_b_left.plot(gamma_value4, Pedge_06_2_150, marker="^", markersize=8, color='red')
ax_b_left.plot(gamma_value3, Pedge_06_1_300, marker="s", markersize=5, color='blue')
ax_b_left.plot(gamma_value4, Pedge_06_2_300, marker="s", markersize=5, color='blue')
ax_b_left.set_xlim(-0.1, 1)
ax_b_left.set_xticks([0, 1])
ax_b_left.set_yticks([0, 0.05])
ax_b_left.tick_params(labelsize=20)
ax_b_right = fig.add_subplot(gs_b[1], sharey=ax_b_left)
ax_b_right.plot(gamma_value5, Pedge_06_3_100, marker="o", markersize=5.5, color='k', label=r'$L=100$')
ax_b_right.plot(gamma_value5, Pedge_06_3_150, marker="^", markersize=8, color='red', label=r'$L=150$')
ax_b_right.plot(gamma_value5, Pedge_06_3_300, marker="s", markersize=6, color='blue', label=r'$L=300$')
ax_b_right.tick_params(left=False, labelleft=False)
ax_b_right.set_xlim(1, 30)
ax_b_right.set_xticks([10, 20, 30])
plt.setp(ax_b_right.get_yticklabels(), visible=False)
ax_b_right.legend(frameon=False, fontsize=20, loc='lower right')
ax_b_right.tick_params(labelsize=20)
fig.text(0.77, 0.04, r'$\gamma$', fontsize=22)
plt.tight_layout()
#plt.savefig('fig6(a-b).png', dpi=400, bbox_inches='tight')
plt.show()
