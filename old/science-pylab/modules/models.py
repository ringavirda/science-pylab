import numpy as np


# f(x) = 0.0000005 * x^2
def exponential1(n: int = 10000, c: int = 0.0000005) -> np.ndarray:
    model = np.zeros((n))
    for i in range(n):
        model[i] = c * i * i
    return model


# f(x) = 1.2*x + 0.1*e^(x*0.011)
def exponential2(n: int = 10000, c1=0.1, c2=0.011) -> np.ndarray:
    model = np.zeros((n))
    for i in range(n):
        model[i] = 30 + 0.01*i + c1 * np.exp(i * c2)
    return model


# f(x) = 0.2*sin(0.005*x) + 2.12*cos(0.005*x)
def transcendental1(n: int = 1000) -> np.ndarray:
    model = np.zeros((n))
    for i in range(n):
        model[i] = 0.2 * np.sin(0.005 * i) + 2.12 * np.cos(0.005 * i)
    return model


# f(x) = 21.5*sin(0.1*x) - 12.3*cos(0.1*x)
def transcendental2(n: int = 100) -> np.ndarray:
    model = np.zeros((n))
    for i in range(n):
        model[i] = 21.5 * np.sin(0.1 * i) - 12.3 * np.cos(0.1 * i)
    return model


# f(x) = -7.22*cos(0.015*x) - 0.54*sin(0.015*x) + 1.9*cos(0.015*x)
def transcendental3(n: int = 500) -> np.ndarray:
    model = np.zeros((n))
    for i in range(n):
        model[i] = (
            -7.22 * np.cos(0.015 * i)
            - 0.54 * np.sin(0.015 * i)
            + 1.9 * np.cos(0.015 * i)
        )
    return model


# f(x) = 30 + 0.005 * x + 4 * sin(0.002 * x) + 7.5 * cos(0.01 * x)
def combined1(n: int = 2000) -> np.ndarray:
    model = np.zeros((n))
    for i in range(n):
        model[i] = 30 + 0.05 * i + 14 * np.sin(0.002 * i) + 17.5 * np.cos(0.01 * i)
    return model


def apply_noise(
    model: np.ndarray,
    mu: float = 0.0,
    sigma: float = 5.0,
    abnorm: bool = False,
    coeff: float = 3.0,
    dens: float = 10.0,
) -> np.ndarray:
    model = model.copy()
    # Normal deviations
    errors = np.random.normal(mu, sigma, model.size)
    with np.nditer(model, op_flags=["readwrite"], flags=["f_index"]) as it:
        for value in it:
            value[...] += errors[it.index]
    if abnorm:
        # Abnormal deviations
        abnormal_count = int((model.size * dens) / 100)
        abnormal_pos = np.zeros(abnormal_count)
        # Fill in positions using normal distribution
        with np.nditer(abnormal_pos, op_flags=["readwrite"]) as it:
            for value in it:
                value[...] = np.ceil(np.random.randint(0, model.size))
        # Fill in abnormals
        abnormals = np.random.normal(mu, sigma * coeff, abnormal_count)
        with np.nditer(abnormal_pos, op_flags=["readwrite"], flags=["f_index"]) as it:
            for value in it:
                model[int(value)] += abnormals[it.index]

    return model
