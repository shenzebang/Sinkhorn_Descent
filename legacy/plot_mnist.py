import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import os

script_path = os.path.dirname(os.path.abspath(__file__))


plot_save_path = os.path.join(script_path, 'barycenter', 'mnist')
SD_nparticles= [2500]
FS_nparticles= [2500]
for digit in range(10):
    # ==============================================================================================================
    #   load the file containing the SD results
    # ==============================================================================================================
    SD_loss = []
    SD_iter = []
    n_SD = len(SD_nparticles)
    for SD_nparticle in SD_nparticles:
        data = np.loadtxt(os.path.join(plot_save_path, "SD/SD_MNIST_digit_{}_nparticle{}".format(digit, SD_nparticle)))
        SD_iter.append(data[0])
        SD_loss.append(data[1])
    # ==============================================================================================================
    #   load the file containing the FS results
    # ==============================================================================================================
    FS_loss = []
    FS_iter = []
    n_FS = len(FS_nparticles)
    for FS_nparticle in FS_nparticles:
        data = np.loadtxt(os.path.join(plot_save_path, "pot_free_support/SD_MNIST_digit_{}_nparticle{}".format(digit, FS_nparticle)))
        FS_iter.append(data[0])
        FS_loss.append(data[1])

    # ==============================================================================================================
    #   plot the results
    # ==============================================================================================================

    fig = plt.figure()
    ax = plt.axes()
    SD_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(n_SD):
        plt.plot(SD_iter[i][1:50], SD_loss[i][1:70], '--{}'.format(SD_colors[i]), label='SD', linewidth=3.0)

    for i in range(n_FS):
        plt.plot(FS_iter[i][1:70], FS_loss[i][1:70], '-k', label='FS', linewidth=3.0)


    plt.xlabel('number of iterations', fontsize=18)
    plt.ylabel('log(Sinkhorn barycenter loss)', fontsize=18)
    plt.yscale('log')
    leg = plt.legend()
    leg_texts = leg.get_texts()
    plt.setp(leg_texts, fontsize='x-large')
    plt.savefig(os.path.join(plot_save_path, 'mnist_digit_{}.pdf'.format(digit)))
    # plt.show()

    print('done')