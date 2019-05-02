import numpy as np
import matplotlib.pyplot as plt


def plot():
    # Time
    T = 10
    # baselines
    # Passive, full feedback
    passive_full_feedback = np.random.rand(T)
    # Passive, binary feedback
    passive_binary_feedback = np.random.rand(T)
    # Active only on full vs none
    active_only_on_full_vs_none = np.random.rand(T)
    # Active on everything <-- full system
    full_system = np.random.rand(T)
    # Random meta controller on all three feedback
    random_meta_controller_on_all = np.random.rand(T)
    # Possible heuristics: first X% of data get full, next (1-X)% get binary, then none
    heurestic = np.random.rand(T)
    x = np.arange(T)
    plt.plot(x, passive_full_feedback)
    plt.plot(x, passive_binary_feedback)
    plt.plot(x, active_only_on_full_vs_none)
    plt.plot(x, full_system)
    plt.plot(x, random_meta_controller_on_all)
    plt.plot(x, heurestic)
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.show()
    print('done plotting')


def main():
    plot()


if __name__ == '__main__':
    main()
