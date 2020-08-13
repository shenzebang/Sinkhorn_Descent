import matplotlib.pyplot as plt
import csv

SMALL_SIZE = 10
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# x = []
# y = []
# z = []
#
# with open('run-.-tag-loss_blur_0.1_lr_0.005_sg_loss.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     i = 0
#     for row in plots:
#         if i is 0:
#             i += 1
#             continue
#         x.append(row[0])
#         y.append(int(row[1]))
#         z.append(float(row[2]))
# # fig = plt.figure()
# # ax = fig.add_subplot(2, 1, 1)
# plt.plot(y[1:100], z[1:100], label='adam')
# # ax.set_yscale('log')
# # plt.yscale('log')
# # plt.xlabel('x')
# # plt.ylabel('y')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()

# x = []
# y = []
# z = []
#
# with open('run-.-tag-generative_model_loss_blur_0.1_constraint_blur_0.1_step_1_jko_steps_40_ng_loss.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     i = 0
#     for row in plots:
#         if i is 0:
#             i += 1
#             continue
#         x.append(row[0])
#         y.append(float(row[1])/40)
#         z.append(float(row[2]))
# # fig = plt.figure()
# # ax = fig.add_subplot(2, 1, 1)
# plt.plot(y[1:], z[1:], label='SiNG')
# # ax.set_yscale('log')
# # plt.yscale('log')
# # plt.xlabel('x')
# # plt.ylabel('y')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()

# x = []
# y = []
# z = []
#
# with open('run-.-tag-GAN_loss_blur_10_constraint_blur_1_step_1_jko_steps_20_ng_loss.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     i = 0
#     for row in plots:
#         if i is 0:
#             i += 1
#             continue
#         x.append(row[0])
#         y.append(float(row[1])/40)
#         z.append(float(row[2]))
# # fig = plt.figure()
# # ax = fig.add_subplot(2, 1, 1)
# plt.plot(y[1:], z[1:], label='SiNG')
# # ax.set_yscale('log')
# plt.yscale('log')
# plt.xlabel('\# epochs')
# plt.ylabel('generator loss')
# plt.title('adversarial ground cost')
# plt.legend()
# plt.savefig('plot/generator_loss.png', bbox_inches='tight')


sgd_time = []
sgd_loss = []
sing_time = []
sing_loss = []

with open('sgd_time.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in plots:
        if i is 0:
            i += 1
            continue
        sgd_time.append(float(row[2]))
with open('sgd_loss.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in plots:
        if i is 0:
            i += 1
            continue
        sgd_loss.append(float(row[2]))

with open('sing_time.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in plots:
        if i is 0:
            i += 1
            continue
        sing_time.append(float(row[2]))
with open('sing_loss.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in plots:
        if i is 0:
            i += 1
            continue
        sing_loss.append(float(row[2]))
# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)
plt.plot(sing_time[1:], sing_loss[1:], label='SiNG')
plt.plot(sgd_time[1:], sgd_loss[1:], label='Adam')
# ax.set_yscale('log')
# plt.yscale('log')
plt.xlabel('wall-clock time/seconds')
plt.ylabel('generator loss')
plt.title('fixed ground metric')
plt.legend()
plt.savefig('plot/generator_loss_fix.png', bbox_inches='tight')

plt.show()


