import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import random

# create list of ground truth vectors [x, y, vx, vy]
# control is a list of [ax, ay] for each timestep
def generate_ground_truth_2D(initial_state, control, num_measurements):
    truths = []
    x = initial_state[0] + random.randrange(-20, 21)
    y = initial_state[1] + random.randrange(-20, 21)
    vx = initial_state[2] + random.randrange(-7, 8)
    vy = initial_state[3] + random.randrange(-7, 8)
    for i in range(num_measurements):
        new_x = x
        new_y = y
        new_vx = vx
        new_vy = vy
        truths.append([new_x, new_y, new_vx, new_vy])
        x = new_x + new_vx + (1.0/2.0)*control[i][0]
        y = new_y + new_vy + (1.0/2.0)*control[i][1]
        vx = new_vx + control[i][0]
        vy = new_vy + control[i][1]
    return np.array(truths)

# create list of measurement vectors [x, y, vx, vy]
def generate_measurements_2D(true_vals, errors):
    measurements = []
    for i in range(len(true_vals)):
        x_measurement = true_vals[i][0] + random.randrange(-int(errors[0]), int(errors[0])+1)
        y_measurement = true_vals[i][1] + random.randrange(-int(errors[1]), int(errors[1])+1)
        vx_measurement = true_vals[i][2] + random.randrange(-int(errors[2]), int(errors[2])+1)
        vy_measurement = true_vals[i][3] + random.randrange(-int(errors[3]), int(errors[3])+1)
        measurements.append([x_measurement, y_measurement, vx_measurement, vy_measurement])
    return np.array(measurements)

def create_covariance_ellipses(covariances, x_estimates, y_estimates):
    ellipses = []
    print(len(x_estimates), len(covariances))
    for i in range(len(x_estimates)):
        cov = covariances[i]
        # just get the x and y covariance, upper left quadrant
        cov = cov[np.ix_([0,3],[0,3])]
        # print("covariance of x and y ",i,":\n", cov)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        theta = np.linspace(0, 2*np.pi, 1000)
        # print((np.sqrt(eigenvalues[None,:]) * eigenvectors))
        ellipsis = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
        ellipsis[0,:] += x_estimates[i]
        ellipsis[1,:] += y_estimates[i]
        ellipses.append((ellipsis[0,:], ellipsis[1,:]))
    return ellipses

# show graph of kalman data
def show_graph(kalman_2d):
    # create figure
    fig, axs = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(8)
    plt.subplots_adjust(bottom=0.25,left=0.3)

    # plots
    x_truths = kalman_2d.truths.T[0]
    y_truths = kalman_2d.truths.T[1]
    truths, = axs.plot(x_truths, y_truths, label="truths", color="red", marker="x", markersize=4)

    x_predictions = kalman_2d.predictions[0]
    y_predictions = kalman_2d.predictions[1]
    pred, = axs.plot(x_predictions, y_predictions, label="predictions", color="blue", marker="o", markersize=4)

    x_measurements = kalman_2d.measurements.T[0]
    y_measurements = kalman_2d.measurements.T[1]
    meas, = axs.plot(x_measurements, y_measurements, label="measurements", color="orange", marker="*", markersize=4)

    x_estimates = kalman_2d.estimations[0]
    y_estimates = kalman_2d.estimations[1]
    ests, = axs.plot(x_estimates, y_estimates, label="estimates", color="green", marker="d", markersize=4)
    axs.legend(loc="upper left")

    # ellipses
    covariance_ellipses = create_covariance_ellipses(kalman_2d.covariances, x_estimates, y_estimates)
    cov_plots = []
    for ellipsis in covariance_ellipses:
        cov_plots.append(axs.plot(ellipsis[0], ellipsis[1], color="green"))

    # sliders
    show_measurements_slider = Slider(plt.axes([0.25, 0.12, 0.65, 0.03]), 'Show Measurements:', 1, kalman_2d.num_measurements+1, valfmt='%0.0f', valstep=1, valinit=kalman_2d.num_measurements)
    show_measurements_slider.on_changed(lambda val: show_measurements_count(int(val), truths, pred, meas, ests, kalman_2d, cov_plots, axs))

    ## CHECKBOXES
    lines = [pred, meas, ests, truths]
    labels = ["Predictions", "Measurements", "Estimates", "True Values"]
    label_vals = [True, True, True, True]
    plot_buttons = CheckButtons(plt.axes([0.05, 0.4, 0.15, 0.30]), labels, label_vals)
    plot_buttons.on_clicked(lambda val: toggle_line(val, lines, labels, fig))

    # labels
    axs.set_xlabel("meters")
    axs.set_ylabel("meters")

    # SHOW
    plt.show()

def show_measurements_count(val, truths, pred, meas, ests, kalman_2d, covs, axs):
    # print(val)
    endpoint = val
    truths.set_xdata(kalman_2d.truths.T[0][:endpoint])
    truths.set_ydata(kalman_2d.truths.T[1][:endpoint])
    pred.set_xdata(kalman_2d.predictions[0][:endpoint])
    pred.set_ydata(kalman_2d.predictions[1][:endpoint])
    meas.set_xdata(kalman_2d.measurements.T[0][:endpoint])
    meas.set_ydata(kalman_2d.measurements.T[1][:endpoint])
    ests.set_xdata(kalman_2d.estimations[0][:endpoint])
    ests.set_ydata(kalman_2d.estimations[1][:endpoint])
    for covariance in covs:
        l = covariance.pop(0)
        l.remove()
    covs.clear()
    ellipses = create_covariance_ellipses(kalman_2d.covariances, kalman_2d.estimations[0][:endpoint], kalman_2d.estimations[1][:endpoint])
    for ellipsis in ellipses:
        covs.append(axs.plot(ellipsis[0], ellipsis[1], color="green"))
    axs.relim()
    axs.autoscale_view()

def toggle_line(val, lines, labels, fig):
    index = labels.index(val)
    lines[index].set_visible(not lines[index].get_visible())
    fig.canvas.draw()