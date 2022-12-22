import seaborn as sb
import random
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 0.1 
s = np.random.normal(mu, sigma, 100)

mu1, sigma1 = 0.5, 1
t = np.random.normal(mu1, sigma1, 100)

sb.lineplot(data= s, color = "red")
sb.lineplot(data= t, color ="blue")
path = "./data/output/{}/learning_curve_{}".format(dataset_name, scoring)
plt.savefig(path)
plt.show()