import matplotlib.pyplot as plt

def plot_accelerometer(accel_data, y_label="Accelerometer", x_label="Time", x_lim=2*10):
    plt.plot(accel_data)
    plt.xlim(0, x_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(['x-axis', 'y-axis', 'z-axis'])
    plt.show()