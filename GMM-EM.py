import numpy as np
from scipy.stats import norm


def GMM_EM(y, K, iteration=100):
    N = len(y)
    alpha = np.ones(K) / K
    mu = np.array(range(0, K))
    sigma = np.ones(K)
    gamma = np.empty((N, K))
    step = 0
    while step < iteration:
        step += 1
        # E step
        for j in range(N):
            for k in range(K):
                gamma[j, k] = alpha[k] * norm.pdf(y[j], mu[k], sigma[k]) / np.dot(alpha, norm.pdf(y[j], mu, sigma))
        # M step
        for k in range(K):
            sigma[k] = np.sqrt(np.dot(gamma[:, k], (y - mu[k]) ** 2) / sum(gamma[:, k]))
            mu[k] = np.dot(gamma[:, k], y) / sum(gamma[:, k])
            alpha[k] = sum(gamma[:, k]) / N

    return mu, sigma, alpha


def simulation(mu, sigma, alpha, N=500):
    K = len(mu)
    y = np.array([])
    for k in range(K):
        y = np.hstack((y, np.random.normal(mu[k], sigma[k], int(N * alpha[k]))))
    np.random.shuffle(y)
    return y


if __name__ == '__main__':
    mu = [10, 30]
    sigma = [1, 2]
    alpha = [0.3, 0.7]
    np.random.seed(1234)
    y = simulation(mu, sigma, alpha)
    mu_est, sigma_est, alpha_est = GMM_EM(y, 2)
    print(f"Estimation  mu: {mu_est} sigma: {sigma_est} alpha: {alpha_est}")
