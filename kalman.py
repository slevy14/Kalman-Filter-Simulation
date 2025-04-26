import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
import random

### INITIALIZE

# initial state
x0 = 1000
vx0 = 40
mu_prev = np.array([x0, vx0]).T.reshape(2, 1)

# initial conditions
ax = 1
dt = 1

# control vector
u = np.array([[ax]])

# observation errors
err_x = 10
err_vx = 1
del_t = 0  # pretend it's 0 for now
Q = np.array([[err_x**2, 0], [0, err_vx**2]])

# augmentation matrices
A = np.array([[1, dt], [0, 1]])
B = np.array([[(1/2)*(dt**2)], [dt]])
C = np.array([[1, 0], [0, 1]])
w = 0  # pretend it's 0 for now

# initialize process covariance
p_err_x = 20
p_err_vx = 5
P_prev = np.array([[p_err_x**2, 0],[0, p_err_vx**2]])
R = np.array([[0, 0], [0, 0]])  # pretend it's 0 for now

## TEST PRINT MATRICES
print("A:\n", A, "\n")
print("B: \n", B, "\n")
print("P_prev: \n", P_prev, "\n")

### GENERATE ALL MEASUREMENTS IN ADVANCE

# create list of ground truth vectors [x, vx]
def generate_ground_truth(x_init, vx_init, num_measurements):
    truths = []
    x = x_init
    vx = vx_init
    for i in range(num_measurements):
        new_x = x
        new_vx = vx
        truths.append([new_x, new_vx])
        x = new_x + new_vx + (1.0/2.0)*ax
        vx = new_vx + ax
    return np.array(truths)

print("TRUTHS: ", generate_ground_truth(4000, 280, 5))
    
# create list of measurement vectors [x, vx]
def generate_measurements(true_vals, x_err, vx_err):
    x_err = int(x_err)
    vx_err = int(vx_err)
    measurements = []
    for i in range(len(true_vals)):
        x_measurement = true_vals[i][0] + random.randrange(-x_err, x_err+1)
        vx_measurement = true_vals[i][1] + random.randrange(-vx_err,vx_err+1)
        measurements.append([x_measurement, vx_measurement])
    return np.array(measurements)

print("MEASUREMENTS: ", generate_measurements(generate_ground_truth(4000, 280, 5), 10, 1))



### RUN ALGORITHM
NUM_MEASUREMENTS = 20

# create empty list of estimation vectors
estimations = np.array([[x0], [vx0]])
predictions = np.array([[x0], [vx0]])
truths = generate_ground_truth(x0, vx0, NUM_MEASUREMENTS)
measurements = generate_measurements(truths, err_x, err_vx)
print(measurements)

# initialize prev
mu_prev = np.array([[x0], [vx0]])
P_prev = np.array([[p_err_x**2, 0],[0, p_err_vx**2]])

# for each measurement:
for i in range(len(measurements)):
    # predict state
    mu_p = np.matmul(A, mu_prev) + np.matmul(B, u) + w
    predictions = np.hstack((predictions, mu_p))
    # predict covariance
    P_p = np.matmul(np.matmul(A, P_prev), np.transpose(A)) + R
    # calculate kalman gain
    K = np.matmul(np.matmul(P_p, np.transpose(C)), np.linalg.inv(np.matmul(np.matmul(C, P_p), np.transpose(C)) + Q))
    # get measurement
    z = (np.matmul(C, measurements[i]) + del_t).T.reshape(2,1)
    # compute state estimation
    mu_t = mu_p + np.matmul(K, (z - np.matmul(C, mu_p)))
    # compute covariance estimation
    P_t = np.matmul((np.identity(len(K)) - np.matmul(K, C)), P_p)
    # save estimation
    # estimations = np.append(estimations, np.array([mu_t]))
    estimations = np.hstack((estimations, mu_t))
    # set current to prev
    mu_prev = mu_t
    P_prev = P_t

## TEST PRINT RESULTS
print(estimations)


### ALGORITHM AS FUNCTION

def kalman_filter(num_measurements, x0, vx0, err_x, err_vx, ax, observed=None):
    # create empty list of estimation vectors
    estimations = np.array([[x0], [vx0]])
    predictions = np.array([[x0], [vx0]])
    truths = generate_ground_truth(x0 + random.randrange(-500,500), vx0 + random.randrange(-20, 20), num_measurements) if observed is None else observed[0]
    measurements = generate_measurements(truths, err_x, err_vx) if observed is None else observed[1]
    covariances = [np.array([[p_err_x**2, 0],[0, p_err_vx**2]])]
    

    # set control vector
    u = np.array([[ax]])

    # initialize prev
    mu_prev = np.array([[x0], [vx0]])
    P_prev = np.array([[p_err_x**2, 0],[0, p_err_vx**2]])

    # for each measurement:
    for i in range(len(measurements)):
        # predict state
        mu_p = np.matmul(A, mu_prev) + np.matmul(B, u) + w
        predictions = np.hstack((predictions, mu_p))
        # predict covariance
        P_p = np.matmul(np.matmul(A, P_prev), np.transpose(A)) + R
        # calculate kalman gain
        K = np.matmul(np.matmul(P_p, np.transpose(C)), np.linalg.inv(np.matmul(np.matmul(C, P_p), np.transpose(C)) + Q))
        # get measurement
        z = (np.matmul(C, measurements[i]) + del_t).T.reshape(2,1)
        # compute state estimation
        mu_t = mu_p + np.matmul(K, (z - np.matmul(C, mu_p)))
        # compute covariance estimation
        P_t = np.matmul((np.identity(len(K)) - np.matmul(K, C)), P_p)
        covariances.append(P_t)
        # save estimation
        # estimations = np.append(estimations, np.array([mu_t]))
        estimations = np.hstack((estimations, mu_t))
        # set current to prev
        mu_prev = mu_t
        P_prev = P_t

    return predictions, measurements, estimations, truths, covariances

def compute_covariance_ellipses(covariances, timesteps, x_estimates):
    ellipses = []
    for i in range(len(timesteps)):
        cov = covariances[i]
        print("covariance",i,":\n", cov)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        theta = np.linspace(0, 2*np.pi, 1000)
        ellipsis = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
        ellipsis[0,:] += timesteps[i]
        ellipsis[1,:] += x_estimates[i]
        ellipses.append((ellipsis[0,:], ellipsis[1,:]))
    return ellipses



# test
results = kalman_filter(20, 1000, 40, err_x, err_vx, ax)
# print(results)

### X VAL

num_measurements = 20
x0 = 400
err_x = 10
vx0 = 20
err_vx = 1
ax = 1

def update_num_measurements(val):
    global num_measurements
    num_measurements = int(val)
    update_graph(True)

def update_x0(val):
    global x0
    x0 = val
    update_graph(False)

def update_err_x(val):
    global err_x
    err_x = val
    update_graph(False)

def update_vx0(val):
    global vx0
    vx0 = val
    update_graph(False)

def update_err_vx(val):
    global err_vx
    err_vx = val
    update_graph(False)

def update_ax(val):
    global ax
    ax = val
    update_graph(False)

def update_graph(regenerate):
    global x_observed
    kalman_results = kalman_filter(num_measurements, x0, vx0, err_x, err_vx, ax, observed=x_observed if not regenerate else None)
    x_observed = (kalman_results[3], kalman_results[1])
    timesteps = [i for i in range(num_measurements)]
    pred.set_ydata(kalman_results[0][0][1:])
    pred.set_xdata(timesteps)
    meas.set_ydata(kalman_results[1].T[0])
    meas.set_xdata(timesteps)
    ests.set_ydata(kalman_results[2][0][1:])
    ests.set_xdata(timesteps)
    truth.set_ydata(kalman_results[3].T[0])
    truth.set_xdata(timesteps)
    for covariance in covs:
        l = covariance.pop(0)
        l.remove()
    covs.clear()
    ellipses = compute_covariance_ellipses(kalman_results[4][1:], timesteps, kalman_results[2][0][1:])
    for ellipsis in ellipses:
        covs.append(axs.plot(ellipsis[0], ellipsis[1], color="green"))
    axs.relim()
    axs.autoscale_view()
    fig.canvas.draw_idle()

def regenerate_measurements(val):
    update_graph(True)

kalman_results = kalman_filter(num_measurements, x0, vx0, err_x, err_vx, ax)

x_predictions = kalman_results[0][0][1:]
x_measurements = kalman_results[1].T[0]
x_estimates = kalman_results[2][0][1:]
x_truths = kalman_results[3].T[0]
x_covariances = kalman_results[4][1:]
x_observed = (kalman_results[3], kalman_results[1])  # storing this for reuse
timesteps = [i for i in range(num_measurements)]

fig, axs = plt.subplots()
plt.subplots_adjust(bottom=0.25,left=0.3)

## SLIDERS
# ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
num_measurements_slider = Slider(plt.axes([0.25, 0.12, 0.65, 0.03]), 'num_measurements', 5, 40)
x0_slider = Slider(plt.axes([0.25, 0.09, 0.65, 0.03]), 'x0', 10, 1000)
err_x_slider = Slider(plt.axes([0.25, 0.06, 0.65, 0.03]), 'err_x', 1, 40)
vx0_slider = Slider(plt.axes([0.25, 0.03, 0.65, 0.03]), 'vx0', 1, 60)
err_vx_slider = Slider(plt.axes([0.25, 0, 0.65, 0.03]), 'err_vx', 1, 40)
ax_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'ax', 1, 20)


num_measurements_slider.on_changed(update_num_measurements)
x0_slider.on_changed(update_x0)
err_x_slider.on_changed(update_err_x)
vx0_slider.on_changed(update_vx0)
err_vx_slider.on_changed(update_err_vx)
ax_slider.on_changed(update_ax)

## BUTTON
measurement_button = Button(plt.axes([0.05, 0.2, 0.15, 0.05]), 'Re-Measure',)
measurement_button.on_clicked(regenerate_measurements)



## PLOT
axs.clear()
pred, = axs.plot(timesteps, x_predictions, label="predictions", color="blue", marker="o", markersize=4)
meas, = axs.plot(timesteps, x_measurements, label="measurements", color="orange", marker="*", markersize=4)
ests, = axs.plot(timesteps, x_estimates, label="estimates", color="green", marker="d", markersize=4)
truth, = axs.plot(timesteps, x_truths, label="true values", color="red", marker="x", markersize=4)
axs.legend(loc="upper left")

## CHECKBOXES
lines = [pred, meas, ests, truth]
labels = ["Predictions", "Measurements", "Estimates", "True Values"]
 
def toggle_line(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    fig.canvas.draw()
 
label = [True, True, True, True]
plot_buttons = CheckButtons(plt.axes([0.05, 0.4, 0.15, 0.30]), labels, label)
plot_buttons.on_clicked(toggle_line)

## COVARIANCE
print("covariances", x_covariances)
covariance_ellipses = compute_covariance_ellipses(x_covariances, timesteps, x_estimates)
covs = []

for ellipsis in covariance_ellipses:
    covs.append(axs.plot(ellipsis[0], ellipsis[1], color="green"))

if __name__ == "__main__":
    plt.show()
else:
    plt.close()