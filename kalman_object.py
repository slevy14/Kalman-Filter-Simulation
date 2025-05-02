import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import random
import analysis_2D


class kalman_object:
    """ MEMBER VARS: (all initialized to None or 0, all public)
    num_measurements
    initial_state_vector (mu)
    control_vector (u)
    process_noise (w)
    A
    B
    C
    initial_P
    process_covariance_noise (R)
    observation_error_list
    Q
    dt
    measurement_noise (del_t)
    truth_function
    measurement_function
    """

    def __init__(self, num_measurements, initial_state_vector, control_vector, dt, observation_error_list, A, B, C, Q, process_noise, process_covariance_noise, initial_P, measurement_noise, truth_function, measurement_function):
        self.num_measurements = num_measurements
        self.initial_state_vector = initial_state_vector
        self.control_vector = control_vector
        self.process_noise = process_noise
        self.dt = dt
        self.observation_error_list = observation_error_list
        self.Q = Q
        self.initial_P = initial_P
        self.process_covariance_noise = process_covariance_noise
        self.A = A
        self.B = B
        self.C = C
        self.measurement_noise = measurement_noise
        self.truth_function = truth_function
        self.measurement_function = measurement_function
        # store last run results
        self.predictions = None
        self.measurements = None
        self.estimations = None
        self.truths = None
        self.covariances = None
        self.full_control = True

    def __str__(self):
        return "PREDICTIONS: \n-------------\n" + str(self.predictions) + "\n-------------\nMEASUREMENTS: \n-------------\n" + str(self.measurements) + "\n-------------\nESTIMATIONS: \n-------------\n" + str(self.estimations) + "\n-------------\nTRUTHS: \n-------------\n" + str(self.truths) + "\n-------------\nCovariances: \n-------------\n" + str(self.covariances)

    
    def run_kalman_filter(self):
        # create empty list of estimation vectors
        estimations = np.array([[x] for x in self.initial_state_vector])
        predictions = np.array([[x] for x in self.initial_state_vector])
        controls = []
        if self.full_control:
            for i in range(self.num_measurements):
                controls.append(self.control_vector)
        else:
            for i in range(self.num_measurements):
                # apply control for first half, then 0 for rest
                if (i < self.num_measurements // 2):
                    controls.append(self.control_vector)
                else:
                    controls.append([0 for x in range(len(self.control_vector))])
        # print(controls)
        truths = self.truth_function(self.initial_state_vector, controls, self.num_measurements)
        measurements = self.measurement_function(truths, self.observation_error_list)
        covariances = [np.array(self.initial_P)]

        # set control vector
        u = np.array([[x] for x in self.control_vector])

        # initialize prev
        mu_prev = np.array([[x] for x in self.initial_state_vector])
        P_prev = self.initial_P

        # for each measurement:
        for i in range(len(measurements)):
            # predict state
            # set u based on controls[i] (this allows acceleration to be removed partway through)
            u = np.array([[x] for x in controls[i]])
            print(u)
            mu_p = (self.A @ mu_prev) + (self.B @ u) + self.process_noise
            predictions = np.hstack((predictions, mu_p))
            # predict covariance
            P_p = (self.A @ P_prev) @ self.A.T + self.process_covariance_noise
            # calculate kalman gain
            K = (P_p @ self.C.T) @ np.linalg.inv(((self.C @ P_p) @ self.C.T) + self.Q)
            # get measurement
            z = (self.C @ measurements[i] + self.measurement_noise).T.reshape(len(self.C),1)
            # compute state estimation
            mu_t = mu_p + (K @ (z - (self.C @ mu_p)))
            # compute covariance estimation
            P_t = (np.identity(len(K)) - (K @ self.C)) @ P_p
            covariances.append(P_t)
            # save estimation
            # estimations = np.append(estimations, np.array([mu_t]))
            estimations = np.hstack((estimations, mu_t))
            # set current to prev
            mu_prev = mu_t
            P_prev = P_t

        self.predictions = predictions
        self.measurements = measurements
        self.estimations = estimations
        self.truths = truths
        self.covariances = covariances
        output_tuple = (predictions, measurements, estimations, truths, covariances)
        return output_tuple
    
