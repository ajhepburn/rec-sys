import numpy as np
import matplotlib.pyplot as plt
 

#  libFM+ALS & 0.586 & 1.166 \\
# libFM+SGD & 0.711 & 0.991 \\
# libFM+SGDA & 0.694 & 0.898 \\
# libFM+MCMC & \textbf{0.737} & \textbf{0.867} \\
# data to plot
n_groups = 4
means_frank = (0.586, 0.694, 0.711, 0.737)
means_guido = (1.166, 0.898, 0.991, 0.867)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.7
 
rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='#914447',
label='AUC')
 
rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='#888374',
label='RMSE')
 
plt.ylabel('Scores')
# plt.xticks(index + bar_width, ('libFM+ALS', 'libFM+SGDA', 'libFM+SGD', 'libFM+MCMC'))
plt.xticks(range(len(means_frank)),('libFM+ALS', 'libFM+SGDA', 'libFM+SGD', 'libFM+MCMC'), rotation=30)

for rect in rects1 + rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % float(height), ha='center', va='bottom')

plt.legend()
 
plt.tight_layout()
plt.show()