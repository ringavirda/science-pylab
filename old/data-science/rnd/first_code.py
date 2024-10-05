# Imports
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

# Configuration
# Experimental measurement quantity
n = 10000
# Experiments count
m = 100
# Abnormals configuration
abnorm_coeff = 3
abnorm_dens = 10
abnorm_count = int((n * abnorm_dens) / 100)
# Gauss config
mu = 0
sigma = 5


# Ideal trend simulation f(x) = 0.0000005 * x^2
def model():
    model = np.zeros((n))
    for i in range(n):
        model[i] = 0.0000005 * i * i
    return model


# Generate normal errors using gauss distribution
def errors_normal(model):
    errors = np.random.normal(mu, sigma, n)
    for i in range(n):
        model[i] += errors[i]
    return model


# Generate abnormal errors
def errors_abnormal(model):
    abnorm_pos = np.zeros((n))

    # Fill in positions using normal distribution
    for i in range(n):
        abnorm_pos[i] = mt.ceil(np.random.randint(1, n))

    # Fill in abnormals
    abnormals = np.random.normal(mu, sigma * abnorm_coeff, abnorm_count)
    for i in range(abnorm_count):
        model[int(abnorm_pos[i])] += abnormals[i]
    return model


# Least squares method
def LSM(model):
    # Prepare data
    Y = np.zeros((n, 1))
    F = np.ones((n, 4))

    # Fill in
    for i, a in np.nditer(np.arange(n)):
        Y[i, 0] = float(model[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)

    start = time.time()
    # Least squares
    F_trans = F.T
    F_squa = F_trans.dot(F)
    F_squa_inv = np.linalg.inv(F_squa)
    F_final = F_squa_inv.dot(F_trans)
    C = F_final.dot(Y)
    approx = F.dot(C)
    execution = time.time() - start
    return (approx, execution)


# RnD transpose to nonlinear (exponent)
def RnD_exponent(model):
    # Prepare data
    approx = np.zeros((n, 1))
    Y = np.zeros((n, 1))
    F = np.ones((n, 4))

    # Fill in
    for i, a in np.nditer(np.arange(n)):
        Y[i, 0] = float(model[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)

    # Least squares
    F_trans = F.T
    F_squa = F_trans.dot(F)
    F_squa_inv = np.linalg.inv(F_squa)
    F_final = F_squa_inv.dot(F_trans)
    C = F_final.dot(Y)

    start = time.time()
    # Extract polynom coeffs
    c0 = C[0, 0]
    c1 = C[1, 0]
    c2 = C[2, 0]
    c3 = C[3, 0]
    # Calculate exponent coeffs
    a0 = c0 - (2 * c2**3) / (9 * c3**2)
    a1 = c1 - (2 * c2**2) / (3 * c3)
    a2 = (2 * c2**3) / (9 * c3**2)
    a3 = (3 * c3) / c2

    # Calculate approximation
    for i in range(n):
        approx[i] = a0 + a1 * i + a2 * mt.exp(a3 * i)
    execution = time.time() - start
    return (approx, execution)


# Iterative exponent method using scipy
def scipy_exponent(model):
    # Declare function for fitting
    def exponent(x, c0, c1, c2, c3):
        return c0 + (c1 * x) + c2 * np.exp(c3 * x)

    # Forward assumptions, guess = 1202.059798705
    guess_c0 = 1
    guess_c1 = 0.01
    guess_c2 = 20
    guess_c3 = 1

    # Prepare data
    vx = np.ones((n))
    vy = np.ones((n))
    for i in np.nditer(np.arange(n)):
        vx[i] = i
        vy[i] = model[i]

    start = time.time()
    # Fit function
    coeffs = curve_fit(exponent, vx, vy, p0=(guess_c0, guess_c1, guess_c2, guess_c3))
    approx = exponent(vx, *coeffs[0])
    execution = time.time() - start
    return (approx, execution)


# Mean squared error
def mse(approx):
    av = np.average(approx)
    r = 0
    for i in range(n):
        r += (approx[i] - av) ** 2
    return np.sum(r) / n


# Linear error
def mle(original, approx):
    r = 0
    for i in range(n):
        r += abs(original[i] - approx[i])
    return np.sum(r) / n


# Plotting
def plot_data(original, data, label):
    plt.clf()
    plt.plot(original)
    plt.plot(data)
    plt.ylabel(label)
    plt.show()


# Main cycle
if __name__ == "__main__":

    # Containers
    mse_results = np.zeros((3, m))
    mle_results = np.zeros((3, m))
    time_results = np.zeros((3, m))

    # Iterative experiments
    for i in range(m):
        # Model creation
        model_ideal = model()
        model_errors = errors_normal(model_ideal)
        model_errors_abnormal = errors_abnormal(model_errors)
        model_fna = model_errors_abnormal

        # Experiments
        lsm_result = LSM(model_fna)
        rnd_result = RnD_exponent(model_fna)
        scipy_result = scipy_exponent(model_fna)

        # Evaluation
        # MSE
        mse_results[0][i] = mse(lsm_result[0])
        mse_results[1][i] = mse(rnd_result[0])
        mse_results[2][i] = mse(scipy_result[0])
        # MLE
        mle_results[0][i] = mle(model_ideal, lsm_result[0])
        mle_results[1][i] = mle(model_ideal, rnd_result[0])
        mle_results[2][i] = mle(model_ideal, scipy_result[0])
        # Time
        time_results[0][i] = lsm_result[1]
        time_results[1][i] = rnd_result[1]
        time_results[2][i] = scipy_result[1]

    diff_rnd_lsm_mse = np.zeros((m))
    diff_rnd_scipy_mse = np.zeros((m))
    diff_rnd_lsm_mle = np.zeros((m))
    diff_rnd_scipy_mle = np.zeros((m))
    for i in range(m):
        diff_rnd_lsm_mse[i] = mse_results[1][i] - mse_results[0][i]
        diff_rnd_scipy_mse[i] = mse_results[1][i] - mse_results[2][i]
        diff_rnd_lsm_mle[i] = mle_results[1][i] - mle_results[0][i]
        diff_rnd_scipy_mle[i] = mle_results[1][i] - mle_results[2][i]

    # Plotting
    # MSE
    plt.clf()
    plt.plot(diff_rnd_lsm_mse, label="RND exp - LSM")
    plt.plot(diff_rnd_scipy_mse, label="RND exp - SCIPY")
    plt.ylabel("MSE results")
    plt.legend()
    plt.savefig("mse.png")
    plt.show()
    # MLE
    plt.clf()
    plt.plot(diff_rnd_lsm_mle, label="RND exp - LSM")
    plt.plot(diff_rnd_scipy_mle, label="RND exp - SCIPY")
    plt.ylabel("MLE results")
    plt.legend()
    plt.savefig("mle.png")
    plt.show()
    # Time
    print(f"LSM mean time: {np.average(time_results[0])}")
    print(f"RND mean time: {np.average(time_results[1])}")
    print(f"SCIPY mean time: {np.average(time_results[2])}")

    # Plotting
    # plot_data(model_fna, lsm_result, "Least Squares")
    # plot_data(model_fna, rnd_result, "RnD Exponent")
    # plot_data(model_fna, iter_result, "Scipy Exponent")

    # Evaluation
    # MSE
    # mse_results = [
    #     mse(model_ideal, lsm_result),
    #     mse(model_ideal, rnd_result),
    #     mse(model_ideal, iter_result)
    # ]
    # MLE
    # mle_results = [
    #     mle(model_ideal, lsm_result),
    #     mle(model_ideal, rnd_result),
    #     mle(model_ideal, iter_result)
    # ]

    # Console echo
    # print("MLE results:")
    # print(mle_results)
    # print("MSE results:")
    # print(mse_results)
