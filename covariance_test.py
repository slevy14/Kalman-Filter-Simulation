import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from kalman import kalman_filter

kalman_results = kalman_filter(12, 10, 5, 3, 0.5, 3)
covariances = kalman_results[4]
# print("covariances", covariances)

colors = []
step = 255.0/len(covariances)
print(step)
for i in range(len(covariances)):
    colors.append((1-step*i/255, 0.8, step*i/255))

for i in range(len(covariances)):
    cov = covariances[i]
    # print("covariance",i,":\n", cov)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    print(eigenvalues,eigenvectors)
    theta = np.linspace(0, 2*np.pi, 1000)
    ellipsis = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
    # print("ellipse", i, " : ", ellipsis)
    plt.plot(ellipsis[0,:], ellipsis[1,:], color=colors[i])

if __name__ == "__main__":
    plt.show()