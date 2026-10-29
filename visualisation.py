import matplotlib.pyplot as plt

def draw_htc_diagram(htc_actual_series, htc_predicted_series = None):
    temp_series = range(0, 860, 10)
    plt.plot(temp_series, htc_actual_series, color='blue', linewidth=2, label='Original')
    plt.xlim(0, 850)
    plt.ylim(0, 12000)
    if htc_predicted_series is not None:
        plt.plot(temp_series, htc_predicted_series, color='green', linewidth=2, label='ANN Result')

    plt.xlabel('Temperature (°C)')
    plt.ylabel('HTC (W/m²K)')
    plt.legend(loc='upper right')
    plt.show()

def draw_loss_curve(results, step):
    plt.plot(results['train_loss'])
    plt.plot(results['test_loss'])
    plt.legend(['train', 'test'])
    plt.ylabel('RMSE')
    plt.xlabel('step')
    plt.savefig(f'./figures/loss_curve_{step}.png')