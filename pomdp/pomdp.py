import sys, getopt
import numpy as np
import random
import time


'''Initializes and runs value iteration on a simple 2d pomdp world'''
class POMDP:
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
        self.T = self.generate_T(y, x, pf, pl, pr, pb, o, g, f)
        self.z = self.generate_z()
        self.utility = np.zeros((self.y_size, self.x_size))
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
        # Initial Policy
        p = self.init_p()

        # Utility vectors
        u = np.zeros(self.x_size * self.y_size)

        # Reward vector
        r = self.generate_r()

        # Test results
        if run_tests:
            self.shortest_paths = self.calculate_paths()
        test_T = None
        if test_probs is not None:
            test_T = self.generate_T(self.y_size, self.x_size, *test_probs, self.obstacles, self.goal, self.fail)
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
                a, v = self.test_pomdp(test_runs, test_steps, test_probs, test_T)
                accuracies_train.append(a)
                values_train.append(v)
                if test_probs is not None:
                    a, v = self.test_pomdp(test_runs, test_steps, test_probs, test_T)
                    accuracies_test.append(a)
                    values_test.append(v)
            # Policy evaluation
            u1 = u.copy()
            u = self.policy_eval(p, u, r, self.T)
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
                    a = self.expected_action(u, self.T, v)
                    if a != p[s]: p[s] = a
        self.utility = u.reshape((self.y_size, self.x_size))
        self.policy = p.reshape((self.y_size, self.x_size))
        if run_tests:
            a, v = self.test_pomdp(test_runs, test_steps, test_probs, test_T)
            accuracies_train.append(a)
            values_train.append(v)
            if test_probs is not None:
                a, v = self.test_pomdp(test_runs, test_steps, test_probs, test_T)
                accuracies_test.append(a)
                values_test.append(v)
        return self.utility, self.policy, accuracies_train, accuracies_test, values_train, values_test

    def test_pomdp(self, runs, steps, probs, test_T):
        distances = []
        values = []
        for _ in range(runs):
            b = self.generate_b()
            s = (0, 0)
            path = [s]
            value = max([self.V(b, a) for a in range(4)])
            for j in range(steps):
                s, b, v = self.take_step(b, s, probs, test_T)
                path.append(s)
                value += v
                if s == self.goal or s == self.fail:
                    break

            values.append(value / (j + 1))
            distances.append(self.shortest_paths[s] + j)
        # self.print_path(path) # For Debugging
        return distances, values

    def take_step(self, b, s, probs, test_T):
        # Calculate value of actions and take max
        values = [self.V(b, a) for a in range(4)]
        max_val = min(values)
        a = values.index(max_val)
        
        # Take step
        s_prime = (0,0)
        T = np.reshape(self.T, (self.y_size, self.x_size, self.y_size, self.x_size, 4))
        if probs is None:
            state_prob = T[s][:, :, a]
            indices = np.arange(self.y_size * self.x_size)
            index = np.random.choice(indices, p=state_prob.reshape(self.y_size * self.x_size))
        else:
            test_T = np.reshape(test_T, (self.y_size, self.x_size, self.y_size, self.x_size, 4))
            state_prob = test_T[s][:, :, a]
            indices = np.arange(self.y_size * self.x_size)
            index = np.random.choice(indices, p=state_prob.reshape(self.y_size * self.x_size))
        s_prime = (int(index / self.y_size), int(index % self.y_size))

        # Make observation
        obs = self.z[s_prime]

        # Adjust belief state
        T_temp = np.zeros((self.y_size, self.x_size, self.y_size, self.x_size, 4))
        T_prime = np.zeros((self.y_size, self.x_size))
        for row in range(self.y_size):
            for col in range(self.x_size):
                T_temp[row, col] = T[row, col] * b[row, col]
        for row in range(self.y_size):
            for col in range(self.x_size):
                T_prime[row, col] = np.sum(T_temp[:, :, row, col])
        b_prime = T_prime * obs
        b_prime = b_prime / np.sum(b_prime)
        return s_prime, b_prime, max_val

    def V(self, b, a):
        T = np.reshape(self.T, (self.y_size, self.x_size, self.y_size, self.x_size, 4))
        T_prime = T[:, :, :, :, a] * np.tile(b, (self.y_size, self.x_size, 1, 1))
        T_prime = np.sum(T_prime, axis=(2,3))
        if np.sum(T_prime) != 0:
            T_prime = T_prime / np.sum(T_prime)

        return np.sum(T_prime * self.utility)

    def generate_b(self):
        return np.full((self.y_size, self.x_size), 1 / (self.y_size * self.x_size))

    def generate_z(self):
        z = np.full((self.y_size, self.x_size, self.y_size, self.x_size), 19, dtype=np.float)
        for row in range(self.y_size):
            for col in range(self.x_size):
                n = row == 0 or (row - 1, col) in self.obstacles
                s = row == self.y_size - 1 or (row + 1, col) in self.obstacles
                e = col == self.x_size - 1 or (row, col + 1) in self.obstacles
                w = col == 0 or (row, col - 1) in self.obstacles
                if (row, col) == self.goal:
                    z[:, :, row, col] = 18
                elif (row, col) == self.fail:
                    z[:, :, row, col] = 17
                elif (row, col) in self.obstacles:
                    z[:, :, row, col] = 16
                elif n and s and e and w:
                    z[:, :, row, col] = 15
                elif n and s and e:
                    z[:, :, row, col] = 14
                elif n and s and w:
                    z[:, :, row, col] = 13
                elif n and e and w:
                    z[:, :, row, col] = 12
                elif s and e and w:
                    z[:, :, row, col] = 11
                elif n and s:
                    z[:, :, row, col] = 10
                elif n and w:
                    z[:, :, row, col] = 9
                elif n and e:
                    z[:, :, row, col] = 8
                elif s and e:
                    z[:, :, row, col] = 7
                elif s and w:
                    z[:, :, row, col] = 6
                elif e and w:
                    z[:, :, row, col] = 5
                elif w:
                    z[:, :, row, col] = 4
                elif e:
                    z[:, :, row, col] = 3
                elif s:
                    z[:, :, row, col] = 2
                elif n:
                    z[:, :, row, col] = 1

        for row in range(self.y_size):
            for col in range(self.x_size):
                obs = z[row, col, row, col]
                z[row, col][z[row,col] != obs] = 0
                count = np.count_nonzero(z[row, col])
                prob = 1 / float(count)
                z[row, col][z[row,col] == obs] = prob
        return z

    def generate_T(self, y_size, x_size, prob_f, prob_l, prob_r, prob_b, obstacles, goal, fail):
        T = np.zeros((x_size * y_size, x_size * y_size, 4))
        counter = 0
        for row in range(1, y_size + 1):
            for col in range(1, x_size + 1):
                up = np.zeros((y_size + 2, x_size + 2))
                left = np.zeros((y_size + 2, x_size + 2))
                down = np.zeros((y_size + 2, x_size + 2))
                right = np.zeros((y_size + 2, x_size + 2))

                if (row - 1, col - 1) not in obstacles and (row - 1, col - 1) != goal and (row - 1, col - 1) != fail:
                    up[row - 1, col] = prob_f
                    up[row, col - 1] = prob_l
                    up[row, col + 1] = prob_r
                    up[row + 1, col] = prob_b
                    left[row, col - 1] = prob_f
                    left[row + 1, col] = prob_l
                    left[row - 1, col] = prob_r
                    left[row, col + 1] = prob_b
                    down[row + 1, col] = prob_f
                    down[row, col + 1] = prob_l
                    down[row, col - 1] = prob_r
                    down[row - 1, col] = prob_b
                    right[row, col + 1] = prob_f
                    right[row - 1, col] = prob_l
                    right[row + 1, col] = prob_r
                    right[row, col - 1] = prob_b

                    for obst in obstacles:
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
                    if col == x_size:
                        right[row, col] += right[row, col + 1]
                        up[row, col] += up[row , col + 1]
                        down[row, col] += down[row, col + 1]
                    if row == 1:
                        up[row, col] += up[row - 1, col]
                        left[row, col] += left[row - 1, col]
                        right[row, col] += right[row - 1, col]
                    if row == y_size:
                        down[row, col] += down[row + 1, col]
                        left[row, col] += left[row + 1, col]
                        right[row, col] += right[row + 1, col]

                up = up[1 : y_size + 1, 1 : x_size + 1]
                left = left[1 : y_size + 1, 1 : x_size + 1]
                down = down[1 : y_size + 1, 1 : x_size + 1]
                right = right[1 : y_size + 1, 1 : x_size + 1]

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

    def print_path(self, path):
        path_matrix = np.zeros((self.y_size, self.x_size))
        for i, step in enumerate(path):
            path_matrix[step] = i

        path_string = ""
        for row in range(self.y_size):
            for col in range(self.x_size):
                if path_matrix[row, col] != 0: path_string += " " + str(path_matrix[row, col]) + "  "
                elif self.policy[row, col] == -1: path_string += " *  "
                elif np.isnan(self.policy[row, col]): path_string += " #  "
                else: path_string += " 0  "
            path_string += '\n'
        print(path_string)
