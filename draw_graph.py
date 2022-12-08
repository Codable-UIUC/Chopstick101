import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pickle

def get_data(cnt: int) -> np.ndarray:
    """
    Get data(image frames) of chosen video

    Parameters
    ----------
    cnt : int; unique number for file and its name

    Returns
    -------
    data : np.ndarray, (2, # of attempted epochs)
    """
    with open(f'./statistics/accuracy_{cnt}','rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    data = []
    cnt = 7
    for num in range(cnt):
        data.append(get_data(num)[1])
    
    x = range(1, 51)
    y = np.average(data, axis=0)
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(1, 50, 500)
    Y_ = X_Y_Spline(X_)
    print()
    ax.plot(X_, Y_, 'o', label=f'trial {cnt}')
    
    ax.set(xlabel='epochs', ylabel='accuracy', title='Accuracy vs. Epochs')
    ax.grid()
    plt.show()