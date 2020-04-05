import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import os

script_path = os.path.dirname(os.path.abspath(__file__))


plot_save_path = os.path.join(script_path, 'plot', 'matching')
SD_nparticles= [2000, 4000]
FW_nparticles= [20100]
# ==============================================================================================================
#   load the file containing the SD results
# ==============================================================================================================
SD_loss = []
SD_iter = []
n_SD = len(SD_nparticles)
for SD_nparticle in SD_nparticles:
    data = np.loadtxt(os.path.join(plot_save_path, "SD_matching_nparticle{}".format(SD_nparticle)))
    SD_iter.append(data[0])
    SD_loss.append(data[1])
# ==============================================================================================================
#   load the file containing the FW results
# ==============================================================================================================
FW_loss = []
FW_iter = []
n_FW = len(FW_nparticles)
for FW_nparticle in FW_nparticles:
    data = np.loadtxt(os.path.join(plot_save_path, "FW_matching_nparticle{}".format(FW_nparticle)))
    FW_iter.append(data[0])
    FW_loss.append(data[1])

# ==============================================================================================================
#   plot the results
# ==============================================================================================================

fig = plt.figure()
ax = plt.axes()
SD_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i in range(n_SD):
    plt.plot(SD_iter[i], SD_loss[i], '--{}'.format(SD_colors[i]), label='SD(N={})'.format(SD_nparticles[i]), linewidth=3.0)

for i in range(n_FW):
    plt.plot(FW_iter[i], FW_loss[i], '-k', label='FW(N={})'.format(FW_nparticles[i]), linewidth=3.0)


plt.xlabel('number of iterations', fontsize=18)
plt.ylabel('log(Sinkhorn barycenter loss)', fontsize=18)
plt.yscale('log')
leg = plt.legend()
leg_texts = leg.get_texts()
plt.setp(leg_texts, fontsize='x-large')
plt.savefig(os.path.join(plot_save_path, 'matching.pdf'))
# plt.show()

print('done')