import numpy as np
import matplotlib.pyplot as plt

import simulation
import tracking
import track_initiation
import visualization

q = 0.05**2
r = 50
measurement_covariance = r*np.identity(2)
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
dt = 2.5
v_max = 25
time = np.arange(0, 25, dt)
F, Q = tracking.DWNAModel.model(dt, q)

true_target_state = np.zeros((4, len(time)))
x0 = np.array([0, 10, 0, 0])
# Generate target trajectory - straight line motion
for k, t in enumerate(time):
    if k == 0:
        true_target_state[:,k] = x0
    else:
        true_target_state[:,k] = F.dot(true_target_state[:,k-1])
# Set up radar
radar_range = 500
clutter_density = 2e-5
P_D = 0.9
P_G = 0.99
radar = simulation.SquareRadar(radar_range, clutter_density, P_D, measurement_covariance)
gate = tracking.TrackGate(P_G, v_max)
target_model = tracking.DWNAModel(q)
PDAF_tracker = tracking.PDAFTracker(P_D, target_model, gate)
N_test = 5
M_req = 3
N_terminate = 5
M_of_N = track_initiation.MOfNInitiation(M_req, N_test, PDAF_tracker, gate)
track_termination = tracking.TrackTerminator(N_terminate)
track_manager = tracking.Manager(PDAF_tracker, M_of_N, track_termination)
measurements_all = []
for k, timestamp in enumerate(time):
    measurements = radar.generate_measurements([H.dot(true_target_state[:,k])], timestamp)
    measurements_all.append(measurements)
    track_manager.step(measurements)
	
fig, ax = visualization.plot_measurements(measurements_all)
ax.plot(true_target_state[2,:], true_target_state[0,:], 'k')
ax.plot(true_target_state[2,0], true_target_state[0,0], 'ko')
visualization.plot_track_pos(track_manager.track_file, ax, 'r')
ax.set_xlim(-radar_range, radar_range)
ax.set_ylim(-radar_range, radar_range)
plt.show()
