import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from matplotlib.lines import Line2D
#=================fig3 data from Lyapunov curve and scattering probability curve.nb======#
# ====================Lefschetz thimble identifying  dominant saddle point.nb=========#
#=====================Fig3b energy spectrum (data).nb===============================#
fig, axs = plt.subplots(2, 2,figsize=(9,8))
path = "C:/Users/26365/Desktop/figure/fig3/"
data = np.loadtxt(path + 'data-fig3a.txt')
x1 = data[0:25, 0]
y1 = data[0:25, 1]
x2 = data[25:2*25, 0]
y2 = data[25:2*25, 1]
x3 = data[25*2:25*3,0]
y3 = data[25*2:25*3,1]
axs[0, 0].plot(x1, y1, color='red', label=r'$t_1=0.2$', linewidth=2)
axs[0, 0].plot(x2, y2, color='blue', label=r'$t_1=0.3$', linewidth=2)
axs[0, 0].plot(x3, y3, color='black', label=r'$t_1=0.4$', linewidth=2)
axs[0,0].set_xlabel(r'$v$', size=20)
axs[0,0].set_ylabel(r'$\lambda(v)$', size=20)
axs[0,0].set_xlim(-0.48,0)
axs[0,0].set_xticks([-0.4,-0.3,-0.2,-0.1,0])
axs[0,0].set_yticks([-0.2, 0])
axs[0,0].set_yticks([-0.2,-0.1,0])
axs[0,0].tick_params(labelsize=15)
axs[0,0].legend(frameon=False, fontsize=15,loc="lower left")
axs[0, 0].axvline(x=-0.2,color="black" ,linestyle='--',ymin=0,ymax=0.95)
axs[0, 0].axvline(x=-0.3, color="black" ,linestyle='--',ymin=0.35,ymax=0.95)
axs[0, 0].axvline(x=-0.4, color="black" ,linestyle='--',ymin=0.35,ymax=0.95)

file_path = "C:/Users/26365/Desktop/figure/fig3/data-fig3b.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()
data = []
for line in lines:
    line = line.strip()
    if line:
        complex_str = line.replace("I", "j").replace(" ", "")
        try:
            complex_number = complex(complex_str)
            data.append(complex_number)
        except ValueError as e:
            print(f"无法解析的字符串: {line}，错误信息: {e}")
data = np.array(data)
real_part1 = np.real(data[0:100])
imag_part1 = np.imag(data[0:100])
real_part2 = np.real(data[100::])
imag_part2 = np.imag(data[100::])
axs[0,1].scatter(real_part2, imag_part2, s=5, color='red', label='_nolegend_')
axs[0,1].scatter(real_part1, imag_part1, s=5, color='black',label='OBC')
axs[0,1].text(0.44, 0.005, r'$\omega_0$', fontsize=15, color='black')
axs[0,1].text(-0.48, 0.005, r'$\omega_0(v=-t_1)$', fontsize=15, color='black')
axs[0,1].set_xlim(-0.8,0.8)
axs[0,1].set_xticks([-0.8,0,0.8])
axs[0,1].set_ylim(-0.4,0.04)
axs[0,1].set_yticks([-0.4,-0.2,0])
axs[0,1].set_xlabel(r'Re(E)', size=20)
axs[0,1].set_ylabel(r'Im(E)', size=20)
axs[0,1].tick_params(labelsize=15)
axs[0,1].axhline(y=0, color='black', linestyle='--', linewidth=2)
pbc_legend = Line2D([0], [0], color='red', lw=2, label='PBC')
handles, labels = axs[0,1].get_legend_handles_labels()
all_handles =  handles+[pbc_legend]
axs[0,1].legend(handles=all_handles, frameon=False, fontsize=14, loc='lower center')

path = "C:/Users/26365/Desktop/figure/fig3/"
data = np.loadtxt(path + 'data-fig3c.txt')
x1 = data[0:25, 0]
y1 = data[0:25, 1]
x2 = data[25:2*25, 0]
y2 = data[25:2*25, 1]
x3 = data[25*2:25*3,0]
y3 = data[25*2:25*3,1]
axs[1, 0].plot(x1, y1, color='lightblue', label=r'$\gamma=0.2$', linewidth=2)
axs[1, 0].plot(x2, y2, color='blue', label=r'$\gamma=0.4$', linewidth=2)
axs[1, 0].plot(x3, y3, color='darkblue', label=r'$\gamma=0.6$', linewidth=2)
axs[1,0].set_xlabel(r'$v$', size=20)
axs[1,0].set_ylabel(r'$\lambda(v)$', size=20)
axs[1,0].set_xlim(0,0.5)
axs[1,0].set_xticks([0,0.1,0.2,0.3,0.4,0.5])
axs[1,0].set_yticks([-3, -2,-1,0])
axs[1,0].tick_params( labelsize=15)
axs[1,0].legend(frameon=False, fontsize=15)

path = "C:/Users/26365/Desktop/figure/fig3/"
data = np.loadtxt(path + 'data-fig3d.txt')
x1 = data[0:80, 0]
y1 = data[0:80, 1]
x2 = data[80:2*80, 0]
y2 = data[80:2*80, 1]
x3 = data[2*80:80*3, 0]
y3 = data[80*2:80*3, 1]
axs[1,1].plot(x1, abs(y1), 'o-', color='red', label=r'$t_1=0.2$', markersize=3, linewidth=1)
axs[1,1].plot(x2, abs(y2), 'o-', color='blue', label=r'$t_1=0.3$', markersize=3, linewidth=1)
axs[1,1].plot(x3, abs(y3), 'o-', color='black', label=r'$t_1=0.4$', markersize=3, linewidth=1)
axs[1,1].set_xlabel(r'$\gamma$',size=20)
axs[1,1].set_ylabel(r'$|\frac{d\lambda(v)}{dv}|_{v=0}|$',size=20)
axs[1,1].set_yticks([0,1.2,2.4])
axs[1,1].set_xlim(0,1.6)
axs[1,1].set_xticks([0,0.4,0.8,1.2,1.6])
axs[1,1].legend(frameon=False,fontsize=14)
axs[1,1].tick_params(labelsize=15)
plt.tight_layout()
plt.show()