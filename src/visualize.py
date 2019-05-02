import numpy as np
import matplotlib.pyplot as plt


def plot():
    x = np.arange(10)
    y = np.arange(10)
    plt.plot(x, y)
    plt.show()
    print('done plotting')


def main():
    plot()


if __name__ == '__main__':
    main()
