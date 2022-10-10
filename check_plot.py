import numpy as np
import scipy.io as scio

import linse_utils as utils
import matplotlib.pyplot as plt


mat_filename = 'matlab/simulation_1.mat'

mat = scio.loadmat(mat_filename)

Jmed_ex_dB = mat['Jmed_ex_dB'].ravel()
J_ex = mat['J_ex'].ravel()

print(Jmed_ex_dB.shape)

fig, ax = plt.subplots()


#ax.plot(Jmed_ex_dB, color=[0.6, 0.6, 0.6], linewidth=0.5)
ax.plot(Jmed_ex_dB, color=[0.6, 0.6, 0.6], linewidth=0.25)
# ax.plot(10*np.log10(J_ex),'k-',linewidth=2)
ax.plot(10*np.log10(J_ex),'k-',linewidth=1)
ax.set_ylabel('EQME (dB)')
ax.set_xlabel('Iterations')
ax.axis([0,J_ex.size,None,None])

fig = utils.format_figure(fig, figsize=(9,4.5))
#fig = utils.format_figure(fig, figsize=(16,10))

fig.savefig('plot_paper(9x4.5).pdf', format='pdf')
fig.savefig('plot_paper(9x4.5).ps', format='ps')
fig.savefig('plot_paper(9x4.5).eps', format='eps', dpi=300)
fig.savefig('plot_paper(9x4.5).png', format='png')
# fig.savefig('plot_screen(16x10).pdf', format='pdf')
# fig.savefig('plot_screen(16x10).png', format='png')
# plt.show()