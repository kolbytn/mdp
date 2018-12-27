import sys, getopt
import numpy as np
import random


'''Initializes and runs value iteration on a simple 2d mdp world'''
class MDP:
    def __init__(self, x, y, o, g, f, pf, pl, pr, pb, gam, eps):
        self.x_size = x
        self.y_size = y
        self.obstacles = o
        self.goal = g
        self.fail = f
        self.prob_f = pf
        self.prob_l = pl
        self.prob_r = pr
        self.prob_b = pb
        self.gamma = gam
        self.epsilon = eps
        self.utility = np.zeros((self.x_size * self.y_size, self.x_size * self.y_size, 4))
        self.policy = np.zeros((self.y_size, self.x_size))
        self.shortest_paths = []

    '''
    Parameters:
        run_test (bool): whether or not to run tests
        test_freq (int): how often to run tests
        test_runs (int): how many tests to run each time
        test_steps (int): max number of steps in a test
        test_probs (1d array): set of action probabilities to also run tests on. Tests not run if None
    Return Values:
        utility (2d numpy array): values for actually being in a state
        policy (2d numpy array): policy representation of utility
        accuracies_train (2d array): accuracy results from testing on training world
        accuracies_test (2d array): accuracy results from testing on given test world
        values_train (2d array): value results from testing on training world 
        values_test (2d array): value results from testing on given test world
    '''
    def value_iter_algorithm(self, run_tests=False, test_freq=1, test_runs=100, test_steps=100, test_probs=None):
        # Transition matrix
        T = self.generate_T()

        # Initial Policy
        p = self.init_p()

        # Utility vectors
        u = np.zeros(self.x_size * self.y_size)

        # Reward vector
        r = self.generate_r()

        # Test results
        if run_tests:
            self.shortest_paths = self.calculate_paths()
        accuracies_train = []
        accuracies_test = []
        values_train = []
        values_test = []

        iteration = 0
        while True:
            iteration += 1
            # Record accuracy
            if run_tests and iteration % test_freq == 0:
                self.policy = p.reshape((self.y_size, self.x_size))
                self.utility = u.reshape((self.y_size, self.x_size))
                a, v = self.test_policy(self.prob_f, self.prob_l, self.prob_r, self.prob_b, test_runs, test_steps)
                accuracies_train.append(a)
                values_train.append(v)
                if test_probs is not None:
                    a, v = self.test_policy(*test_probs, test_runs, test_steps)
                    accuracies_test.append(a)
                    values_test.append(v)
            # Policy evaluation
            u1 = u.copy()
            u = self.policy_eval(p, u, r, T)
            # Stopping criteria
            delta = np.absolute(u - u1).max()
            stop = self.epsilon * (1 - self.gamma) / self.gamma
            if delta < stop: break
            # Print progress
            print("Iteration {}: {:.8f} / {:.8f}, accuracy: {:.2f}, value: {:.2f})"
                .format(iteration,
                        delta,
                        stop,
                        sum(accuracies_train[-1]) / len(accuracies_train[-1]) if len(accuracies_train) > 0 else 0,
                        sum(values_train[-1]) / len(values_train[-1]) if len(values_train) > 0 else 0), end="\r")
            # Update policy
            for s in range(self.y_size * self.x_size):
                if not np.isnan(p[s]) and not p[s] == -1:
                    v = np.zeros((1, self.y_size * self.x_size))
                    v[0, s] = 1.0
                    # Policy improvement
                    a = self.expected_action(u, T, v)
                    if a != p[s]: p[s] = a
        self.utility = u.reshape((self.y_size, self.x_size))
        self.policy = p.reshape((self.y_size, self.x_size))
        if run_tests:
            a, v = self.test_policy(self.prob_f, self.prob_l, self.prob_r, self.prob_b, test_runs, test_steps)
            accuracies_train.append(a)
            values_train.append(v)
            if test_probs is not None:
                a, v = self.test_policy(*test_probs, test_runs, test_steps)
                accuracies_test.append(a)
                values_test.append(v)
        return self.utility, self.policy, accuracies_train, accuracies_test, values_train, values_test

    def test_policy(self, pf, pl, pr, pb, runs, steps):
        distances = []
        values = []
        agent = (0, 0)
        for i in range(runs):
            value = 0
            value += self.utility[agent]
            for j in range(steps):
                temp = (0, 0)
                r = random.random()

                # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
                if self.policy[agent] == 0:
                    if r < pb:
                        temp = tuple(sum(x) for x in zip(agent, (1,0)))
                    elif r < pb + pr:
                        temp = tuple(sum(x) for x in zip(agent, (0,1)))
                    elif r < pb + pr + pl:
                        temp = tuple(sum(x) for x in zip(agent, (0,-1)))
                    else:
                        temp = tuple(sum(x) for x in zip(agent, (-1,0)))
                elif self.policy[agent] == 1:
                    if r < pb:
                        temp = tuple(sum(x) for x in zip(agent, (0,1)))
                    elif r < pb + pr:
                        temp = tuple(sum(x) for x in zip(agent, (-1,0)))
                    elif r < pb + pr + pl:
                        temp = tuple(sum(x) for x in zip(agent, (1,0)))
                    else:
                        temp = tuple(sum(x) for x in zip(agent, (0,-1)))
                elif self.policy[agent] == 2:
                    if r < pb:
                        temp = tuple(sum(x) for x in zip(agent, (-1,0)))
                    elif r < pb + pr:
                        temp = tuple(sum(x) for x in zip(agent, (0,-1)))
                    elif r < pb + pr + pl:
                        temp = tuple(sum(x) for x in zip(agent, (0,1)))
                    else:
                        temp = tuple(sum(x) for x in zip(agent, (1,0)))
                elif self.policy[agent] == 3:
                    if r < pb:
                        temp = tuple(sum(x) for x in zip(agent, (0,-1)))
                    elif r < pb + pr:
                        temp = tuple(sum(x) for x in zip(agent, (1,0)))
                    elif r < pb + pr + pl:
                        temp = tuple(sum(x) for x in zip(agent, (-1,0)))
                    else:
                        temp = tuple(sum(x) for x in zip(agent, (0,1)))

                if temp[0] < self.y_size and temp[0] >= 0 and \
                        temp[1] < self.x_size and temp[1] >= 0 and \
                        self.policy[temp] is not None:
                    agent = temp
                value += self.utility[agent]
                if self.policy[agent] == -1:
                    break

            values.append(value / (j + 1))
            distances.append(self.shortest_paths[agent] + j)
            agent = (0, 0)
        return distances, values

    def calculate_paths(self):
        dist = np.full((self.y_size + 2, self.x_size + 2), -1)
        dist[(self.goal[0] + 1, self.goal[1] + 1)] = 0

        for i in range(self.y_size * self.x_size):
            for row in range(1, self.y_size + 1):
                for col in range(1, self.x_size + 1):
                    if (row - 1, col - 1) not in self.obstacles:
                        n = [dist[row-1,col], dist[row+1,col], dist[row,col+1], dist[row,col-1]]
                        neighbors = list(filter(lambda x: x != -1, n))
                        if len(neighbors) > 0:
                            d = min(neighbors) + 1
                            dist[row, col] = d if d < dist[row, col] or dist[row, col] == -1 else dist[row, col]

        dist[self.fail] = self.y_size * self.x_size - 1
        return dist[1:(self.y_size + 1), 1:(self.x_size + 1)]

    def generate_T(self):
        T = np.zeros((self.x_size * self.y_size, self.x_size * self.y_size, 4))
        counter = 0
        for row in range(1, self.y_size + 1):
            for col in range(1, self.x_size + 1):
                up = np.zeros((self.y_size + 2, self.x_size + 2))
                left = np.zeros((self.y_size + 2, self.x_size + 2))
                down = np.zeros((self.y_size + 2, self.x_size + 2))
                right = np.zeros((self.y_size + 2, self.x_size + 2))

                if (row - 1, col - 1) not in self.obstacles and (row - 1, col - 1) != self.goal and (row - 1, col - 1) != self.fail:
                    up[row - 1, col] = self.prob_f
                    up[row, col - 1] = self.prob_l
                    up[row, col + 1] = self.prob_r
                    up[row + 1, col] = self.prob_b
                    left[row, col - 1] = self.prob_f
                    left[row + 1, col] = self.prob_l
                    left[row - 1, col] = self.prob_r
                    left[row, col + 1] = self.prob_b
                    down[row + 1, col] = self.prob_f
                    down[row, col + 1] = self.prob_l
                    down[row, col - 1] = self.prob_r
                    down[row - 1, col] = self.prob_b
                    right[row, col + 1] = self.prob_f
                    right[row - 1, col] = self.prob_l
                    right[row + 1, col] = self.prob_r
                    right[row, col - 1] = self.prob_b

                    for obst in self.obstacles:
                        up[row, col] += up[tuple(x + 1 for x in obst)]
                        left[row, col] += left[tuple(x + 1 for x in obst)]
                        down[row, col] += down[tuple(x + 1 for x in obst)]
                        right[row, col] += right[tuple(x + 1 for x in obst)]
                        up[tuple(x + 1 for x in obst)] = 0
                        left[tuple(x + 1 for x in obst)] = 0
                        down[tuple(x + 1 for x in obst)] = 0
                        right[tuple(x + 1 for x in obst)] = 0

                    if col == 1:
                        left[row, col] += left[row, col - 1]
                        up[row, col] += up[row, col - 1]
                        down[row, col] += down[row, col - 1]
                    if col == self.x_size:
                        right[row, col] += right[row, col + 1]
                        up[row, col] += up[row , col + 1]
                        down[row, col] += down[row, col + 1]
                    if row == 1:
                        up[row, col] += up[row - 1, col]
                        left[row, col] += left[row - 1, col]
                        right[row, col] += right[row - 1, col]
                    if row == self.y_size:
                        down[row, col] += down[row + 1, col]
                        left[row, col] += left[row + 1, col]
                        right[row, col] += right[row + 1, col]

                up = up[1 : self.y_size + 1, 1 : self.x_size + 1]
                left = left[1 : self.y_size + 1, 1 : self.x_size + 1]
                down = down[1 : self.y_size + 1, 1 : self.x_size + 1]
                right = right[1 : self.y_size + 1, 1 : self.x_size + 1]

                T[counter, :, 0] = up.flatten()
                T[counter, :, 1] = left.flatten()
                T[counter, :, 2] = down.flatten()
                T[counter, :, 3] = right.flatten()

                counter += 1

        return T

    def generate_r(self):
        r = np.full((self.y_size, self.x_size), -.04)
        for obst in self.obstacles:
            r[obst] = 0
        r[self.goal] = 1
        r[self.fail] = -1
        return r.flatten()

    def init_p(self):
        # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
        p = np.random.randint(0, 4, size=(self.y_size, self.x_size)).astype(np.float32)
        for obst in self.obstacles:
            p[obst] = np.NaN
        p[self.goal] = p[self.fail] = -1
        return p.flatten()

    def policy_eval(self, p, u, r, T):
        for s in range(self.y_size * self.x_size):
            if not np.isnan(p[s]):
                v = np.zeros((1, self.y_size * self.x_size))
                v[0,s] = 1.0
                action = int(p[s])
                u[s] = r[s] + self.gamma * np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
        return u

    def expected_action(self, u, T, v):
        actions_array = np.zeros(4)
        for action in range(4):
            # Expected utility of doing a in state s, according to T and u.
            actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
        return np.argmax(actions_array)

    def print_policy(self, p, shape):
        counter = 0
        policy_string = ""
        for row in range(shape[0]):
            for col in range(shape[1]):
                if p[counter] == -1: policy_string += " *  "
                elif p[counter] == 0: policy_string += " ^  "
                elif p[counter] == 1: policy_string += " <  "
                elif p[counter] == 2: policy_string += " v  "
                elif p[counter] == 3: policy_string += " >  "
                elif np.isnan(p[counter]): policy_string += " #  "
                counter += 1
            policy_string += '\n'
        print(policy_string)
