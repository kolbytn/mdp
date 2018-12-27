from pomdp import POMDP
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def main():
    # World features
    x = 5
    y = 5
    o = [(1, 1), (2, 1), (3, 1), (3, 2), (0, 3), (1, 3), (3, 3)]

    g = (4, 4)
    f = (0, 4)

    # Action Probabilities
    pf = .8
    pl = .1
    pr = .1
    pb = 0

    # Hyper Parameters
    gam = .999
    eps = .0001

    # Tests
    test_freq = 5
    test_runs = 10
    test_steps = 30

    # Train and Test Ideal World
    world = POMDP(x, y, o, g, f, pf, pl, pr, pb, gam, eps)

    utility_i, policy_i, dist_train_i, dist_test_i, val_train_i, val_test_i = \
        world.value_iter_algorithm(run_tests=True,
                                   test_freq=test_freq,
                                   test_runs=test_runs,
                                   test_steps=test_steps)

    # Graph Results
    create_figure("POMDP Loss", dist_train_i, test_freq)


def create_figure(name, data, tick_freq):
    if len(data) == 0:
        return
    
    samples = [
        int(len(data) * .25),
        int(len(data) * .5),
        int(len(data) * .75),
        int(len(data) - 1)]

    sampled = [data[x] for x in samples]

    plt.clf()
    plt.plot([sum(x) / len(x) for x in ([data[0]] + sampled)])
    plt.boxplot(sampled)
    plt.xticks(range(len(samples) + 1), [0] + [x * tick_freq for x in samples])
    plt.title(name)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("graphs/" + name + ".png")


if __name__ == "__main__":
    main()
