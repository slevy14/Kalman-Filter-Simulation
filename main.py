import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import random
import kalman_object
import analysis_2D

if __name__ == "__main__":
    # create kalman object (2d)
    # initial state
    num_measurements = 100
    x0 = 0
    y0 = 0
    vx0 = 10
    vy0 = 20
    initial_state = [x0, y0, vx0, vy0]

    # initial conditions
    ax = 1
    ay = -1
    dt = 1

    # control vector
    u = [ax, ay]

    # observation errors
    err_x = 10
    err_y = 10
    err_vx = 1
    err_vy = 1
    observation_error = np.array([err_x, err_y, err_vx, err_vy])
    del_t = 0  # pretend it's 0 for now
    Q = np.array([[err_x**2, 0, 0, 0], [0, err_y**2, 0, 0], [0, 0, err_vx**2, 0], [0, 0, 0, err_vy**2]])

    # augmentation matrices
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[(1/2)*(dt**2), 0], [0, (1/2)*(dt**2)], [dt, 0], [0, dt]])
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    w = 0  # pretend it's 0 for now

    # initialize process covariance
    p_err_x = 20
    p_err_y = 20
    p_err_vx = 5
    p_err_vy = 5
    P_prev = np.array([[p_err_x**2, 0, 0, 0], [0, p_err_y**2, 0, 0], [0, 0, p_err_vx**2, 0], [0, 0, 0, p_err_vy**2]])
    R = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  # pretend it's 0 for now

    # run the kalman filter
    kalman_2d = kalman_object.kalman_object(num_measurements, initial_state, u, dt, observation_error, A, B, C, Q, w, R, P_prev, del_t, analysis_2D.generate_ground_truth_2D, analysis_2D.generate_measurements_2D)
    print(kalman_2d)
    kalman_results = kalman_2d.run_kalman_filter()
    print(kalman_2d)


    ### PLOT
    ## state vector is constructed as [x, y, vx, vy]
    analysis_2D.show_graph(kalman_2d)