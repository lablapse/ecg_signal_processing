import matplotlib.pyplot as plt
import numpy as np
import linse_utils

# Load data
data = np.load('dados.npz')

# Training set
X_train = data['X_train']
y_train = data['y_train']

# Test set
X_test = data['X_test']
y_test = data['y_test']


NORM = X_train[3, ].mean(axis=-1)
MI = X_test[205, ].mean(axis=-1)
CD = X_test[5, ].mean(axis=-1)
STTC = X_train[38, ].mean(axis=-1)
HYP = X_train[23, ].mean(axis=-1)

NORM_CUT = NORM[86:236]
MI_CUT = MI[118:268]
CD_CUT = CD[340:490]
STTC_CUT = STTC[550:700]
HYP_CUT = HYP[280:430]

#print(NORM_CUT.shape)
#print(MI_CUT.shape)


# index = 395 - 53 - 1 #train
# index = 31-1 #teste

# valor_med = X_test[index, ].mean(axis=-1)
# fig_s, ax_s = plt.subplots(figsize=(14,5))
# ax_s.set_title(f'Index = {index}        Rótulo = {y_test[index]}')
# ax_s.plot(valor_med)
# plt.show()

# valor_med = X_train[index, ].mean(axis=-1)
# fig_s, ax_s = plt.subplots(figsize=(14,5))
# ax_s.set_title(f'Index = {index}        Rótulo = {y_train[index]}')
# ax_s.plot(valor_med)
# plt.show()


# NORM: TRAIN - INDEX 3
# MI: TEST - INDEX 205
# CD: TEST - INDEX 5
# STTC: TRAIN - INDEX 38
# HYP: TRAIN - INDEX 23

#NOVOOOOOOOOOOOOOOOOOOOOOOOOO
fig, ax = plt.subplots()
local = 'E:/Usuários/Sarah/Documentos/UTFPR/TCC/Imagens/'

ax.plot(HYP_CUT, color=[0, 0, 0], linewidth=0.25)
ax.set_ylabel('Amplitude')
ax.set_xlabel('Amostras')
ax.axis([0,HYP_CUT.size,None,None])

fig = linse_utils.format_figure(fig, figsize=(9,4.5))
#fig = utils.format_figure(fig, figsize=(16,10))


# fig.savefig(local+'/HYP(9x4.5)_cut.pdf', format='pdf')
# fig.savefig('plot_paper(9x4.5).ps', format='ps')
# fig.savefig('plot_paper(9x4.5).eps', format='eps', dpi=300)
#fig.savefig('NORM PNG(9x4.5).png', format='png')
# fig.savefig('plot_screen(16x10).pdf', format='pdf')
# fig.savefig('plot_screen(16x10).png', format='png')
plt.show()

print('end')



# VELHOOOOOOOOOOOOOOOOO
# plt.rcParams["font.family"] = "Times New Roman"

# NORM = X_train[3, ].mean(axis=-1)
# MI = X_test[205, ].mean(axis=-1)
# CD = X_test[5, ].mean(axis=-1)
# STTC = X_train[38, ].mean(axis=-1)
# HYP = X_train[23, ].mean(axis=-1)

# fig, axs = plt.subplots(nrows=5, ncols=1, constrained_layout=True, figsize=(8,30))
# axs[0].plot(NORM)
# axs[0].set_title('Normal')
# axs[1].plot(MI)
# axs[1].set_title('Infarto do Miocárdio')
# axs[2].plot(CD)
# axs[2].set_title('Distúrbios de Condução')
# axs[3].plot(STTC)
# axs[3].set_title('Alterações ST/T')
# axs[4].plot(HYP)
# axs[4].set_title('Hipertrofia')
# plt.show()

#print('end')