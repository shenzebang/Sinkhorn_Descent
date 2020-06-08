import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import torch
import os

script_path = os.path.dirname(os.path.abspath(__file__))


plot_save_path = os.path.join(script_path, 'plot', 'gan', 'cifar10')
# epoch length 54
# plot only the first 20 epoch
smooth_range = range(16, 16*50)

result_Adam = torch.load("{}/Adam.pt".format(plot_save_path))
result_amsgrad = torch.load("{}/amsgrad.pt".format(plot_save_path))
result_rmsprop = torch.load("{}/rmsprop.pt".format(plot_save_path))
result_SiNG = torch.load("{}/SiNG.pt".format(plot_save_path))

result_Adam_smooth = result_Adam.clone()
result_amsgrad_smooth = result_amsgrad.clone()
result_rmsprop_smooth = result_rmsprop.clone()
result_SiNG_smooth = result_SiNG.clone()
for i in smooth_range:
    result_Adam_smooth[0, i] = min(result_Adam[0, i - 2], result_Adam[0, i - 1], result_Adam[0, i])
    # result_Adam_smooth[1, i] = result_Adam[1, i]
    result_amsgrad_smooth[0, i] = min(result_amsgrad[0, i - 2], result_amsgrad[0, i - 1], result_amsgrad[0, i])
    # result_amsgrad_smooth[1, i] = result_amsgrad[1, i]
    result_rmsprop_smooth[0, i] = min(result_rmsprop[0, i - 2], result_rmsprop[0, i - 1], result_rmsprop[0, i])
    # result_rmsprop_smooth[1, i] = result_rmsprop[1, i]
    result_SiNG_smooth[0, i] = min(result_SiNG[0, i - 2], result_SiNG[0, i - 1], result_SiNG[0, i])
    # result_SiNG_smooth[1, i] = result_SiNG[1, i]
# ==============================================================================================================
#   plot the results
# ==============================================================================================================

fig = plt.figure()
ax = plt.axes()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# for i in range(n_SD):
plt.plot(result_Adam_smooth[1, 0:16*50], result_Adam_smooth[0, 0:16*50], '--{}'.format(colors[0]), label='Adam', linewidth=3.0)
plt.plot(result_amsgrad_smooth[1, 0:16*50], result_amsgrad_smooth[0, 0:16*50], '--{}'.format(colors[1]), label='amsgrad', linewidth=3.0)
plt.plot(result_rmsprop_smooth[1, 0:16*50], result_rmsprop_smooth[0, 0:16*50], '--{}'.format(colors[2]), label='RMSprop', linewidth=3.0)
plt.plot(result_SiNG_smooth[1, 0:16*50], result_SiNG_smooth[0, 0:16*50], '--{}'.format(colors[3]), label='SiNG', linewidth=3.0)
# for i in range(n_FW):
#     plt.plot(FW_iter[i], FW_loss[i], '-k', label='FW(N={})'.format(FW_nparticles[i]), linewidth=3.0)


plt.xlabel('number of epochs', fontsize=18)
plt.ylabel('generator loss', fontsize=18)
plt.yscale("log")
leg = plt.legend()
leg_texts = leg.get_texts()
plt.setp(leg_texts, fontsize='x-large')
plt.savefig(os.path.join(plot_save_path, 'gan_cifar10.pdf'))
# plt.show()

print('done')