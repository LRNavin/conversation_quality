import utilities.data_read_util as reader
import numpy as np
import matplotlib.pyplot as plt

start=0
duration=10
day=2
participant_id=2

test_array = reader.get_accel_data_from_participant_between(day=day, participant_id=participant_id,
                                                            start_time=start, duration=duration)
print(np.shape(test_array))

plt.plot(test_array)
plt.xlim(0,duration*2)
plt.ylabel('ACCELRO')
plt.show()


