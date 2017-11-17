import matplotlib.pyplot as plt
import seaborn as sns


def plot_rewards(fpath, title=None):

    with open(fpath, 'r') as f:
        l = f.readlines()

    i = 0
    for i, x in enumerate(l):

        if 'Timestep' in x:
            break

    l = l[i:]

    tsteps = [int(x.split()[-1].strip()) for x in l[::6]]
    rews = [float(x.split()[-1].strip()) for x in l[1::6]]

    plt.plot(tsteps, rews)

    plt.title(title)

    plt.xlabel('Timesteps')
    plt.ylabel('Mean reward (100 timesteps)')

    plt.show()


if __name__ == '__main__':

    plot_rewards('log.txt', 'Default Configuration on Pong')
