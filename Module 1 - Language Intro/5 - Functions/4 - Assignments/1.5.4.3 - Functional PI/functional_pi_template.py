import math


def my_pi(target_error):
    """
    Implementation of Gauss–Legendre algorithm to approximate PI from https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm

    :param target_error: Desired error for PI estimation
    :return: Approximation of PI to specified error bound
    """

    ### YOUR CODE HERE ###
    a = 1.0
    b = 1.0 / (2 ** (1/2))
    t = 0.25
    p = 1.0

    while True:
        A = a
        B = b
        P = p
        T = t
        a = (a + B)/2
        b = (A * b)**(1/2)
        p = 2 * p
        t = t - P * (a - A)**2
        pi_new = ((a+b)**2)/(4 * t)
        pi_old = ((A+B)**2)/(4 * T)

        if abs(pi_new - pi_old) < target_error:
            return pi_new
    # change this so an actual value is returned
    return 0




desired_error = 1E-10

approximation = my_pi(desired_error)

print("Solution returned PI=", approximation)

error = abs(math.pi - approximation)

if error < abs(desired_error):
    print("Solution is acceptable")
else:
    print("Solution is not acceptable")
