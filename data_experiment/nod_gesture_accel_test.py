import utilities.data_read_util as reader
import utilities.data_plot_util as plotter
import feature_extract.synchrony_extractor as sync_extractor
import numpy as np
import matplotlib.pyplot as plt

start=0
duration=10
day=2
participant_id_1=2
participant_id_2=5
correl_axis=2

individual1_acc = reader.get_accel_data_from_participant_between(day=day, participant_id=participant_id_1,
                                                            start_time=start, duration=duration)[:,correl_axis]
print(np.shape(individual1_acc))
individual2_acc = reader.get_accel_data_from_participant_between(day=day, participant_id=participant_id_2,
                                                            start_time=start, duration=duration)[:,correl_axis]
print(np.shape(individual1_acc))

plotter.plot_accelerometer(accel_data=individual1_acc, x_lim=2*duration)

print(sync_extractor.get_correlation_between(individual1_acc, individual2_acc))


print(sync_extractor.get_mutual_info_between(individual1_acc, individual2_acc, True))

if 1:
    cross_correl = []
    for i in range(0,20):
        curr_correl = sync_extractor.get_timelagged_correl_between(individual1_acc, individual2_acc, i)
        cross_correl.append(curr_correl)
        print(curr_correl)

    plt.plot(cross_correl)
    plt.show()
else:
    print(sync_extractor.get_timelagged_correl_between(individual1_acc, individual2_acc, 0))