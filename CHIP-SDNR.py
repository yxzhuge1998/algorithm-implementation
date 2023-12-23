import copy
import numpy as np
from numpy import ix_


def g(v, l, tau):
    gv = (abs(v) - l) ** 2 * (2 * l * tau ** 2 - (tau + 1) * abs(v)) / (l ** 2 * (tau - 1) ** 3) * np.sign(v)
    return gv


def h(v, l, tau):
    hv = (abs(v) - l) / (l ** 2 * (tau - 1) ** 3) * (3 * (tau + 1) * abs(v) - l * (4 * tau ** 2 + tau + 1))
    return hv


class CHIP_SDNR:
    def __init__(self):
        self.d = None
        self.beta = None
        self.beta_opt = None
        self.lambda_opt = None

    def sdnr(self, X, y, l, tau, K, beta, d):
        n, p = X.shape
        I = np.identity(p)
        zero = np.zeros((p, p))
        G = X.T @ X / n
        ytilde = X.T @ y / n
        A_old = {}
        k = 0
        while k < K:
            A1 = [i for i in range(p) if l < abs(beta[i] + d[i]) < l * tau]
            A2 = [i for i in range(p) if abs(beta[i] + d[i]) >= l * tau]
            A = list(set(A1).union(set(A2)))
            A.sort()
            B = [i for i in range(p) if abs(beta[i] + d[i]) <= l]
            z = np.vstack((beta[ix_(A1, [0])], d[ix_(A2, [0])], beta[ix_(B, [0])],
                           d[ix_(A1, [0])], beta[ix_(A2, [0])], d[ix_(B, [0])]))

            d[ix_(A2, [0])] = 0
            beta[ix_(B, [0])] = 0

            ztildeA1 = beta[ix_(A1, [0])] + d[ix_(A1, [0])]

            M = np.vstack((I[ix_(A1, A1)], zero[ix_(A2, A1)]))

            N = np.hstack((I[ix_(A1, A1)] + np.diag(h(ztildeA1, l, tau).flatten()), zero[ix_(A1, A2)]))

            betaAdA1 = (np.linalg.inv(np.vstack((np.hstack((G[ix_(A, A)], M)),
                                                 np.hstack((N, np.diag(h(z[ix_(A1, [0])], l, tau).flatten()))))))
                        @ np.vstack((ytilde[ix_(A, [0])],
                                     g(ztildeA1, l, tau) + np.diag(h(ztildeA1, l, tau).flatten()) @ ztildeA1)))

            beta[ix_(A, [0])] = betaAdA1[ix_(list(range(0, len(A))), [0])]
            d[ix_(A1, [0])] = betaAdA1[ix_(list(range(len(A), len(betaAdA1))), [0])]
            d[ix_(B, [0])] = ytilde[ix_(B, [0])] - G[ix_(B, A)] @ beta[ix_(A, [0])]

            if set(A) == set(A_old) or k >= K:
                break
            else:
                k = k + 1
                A_old = A

            self.beta = beta

        return self

    def sdnr_continuation(self, X, y, M=10, alpha=0.5, K=20, tau=3.7):
        n, p = X.shape
        self.beta = np.zeros((p, 1))
        self.d = X.T @ y / n
        l0 = np.linalg.norm((X.T @ y / n).flatten(), ord=np.inf)
        hbic_min = np.inf
        for m in range(M):
            l = l0 * alpha ** (m + 1)
            self.sdnr(X, y, l, tau, K, self.beta, self.d)
            hbic = (np.log((np.linalg.norm(y - X @ self.beta)) ** 2 / n) +
                    np.log(np.log(n)) * np.log(p) / n * np.count_nonzero(self.beta))

            if hbic < hbic_min:
                hbic_min = hbic
                self.beta_opt = copy.deepcopy(self.beta)
                self.lambda_opt = l
            if np.linalg.norm(self.beta.flatten(), ord=0) > np.floor(n / np.log(p)):
                break

        return self


def generate_beta(T, p, R):
    theta = np.random.binomial(1, T / p, p)
    kappa = np.random.uniform(0, 1, p)
    beta = theta * R ** kappa
    return beta


def simulation1(n=400, p=4000, rho=0.3, sigma=0.5):
    T = 20
    R = 10

    mean = np.zeros(p)
    covariance_matrix = np.array([rho ** abs(i - j) for i in range(p) for j in range(p)]).reshape(p, p)
    X = np.random.multivariate_normal(mean, covariance_matrix, size=n)
    beta = generate_beta(T, p, R)
    epsilon = np.random.normal(0, sigma, n).reshape(n, 1)
    y = X @ beta.reshape(-1, 1) + epsilon

    return X, y


def simulation2(n=1000, p=10000, rho=0.3, sigma=0.5):
    T = 40
    R = 10

    Xtilde = np.random.randn(n, p)
    X = np.zeros((n, p))
    X[:, 0] = Xtilde[:, 0]
    for i in range(1, p - 1):
        X[:, i] = Xtilde[:, i] + rho * (Xtilde[:, i + 1] + Xtilde[:, i - 1])
    X[:, p - 1] = Xtilde[:, p - 1]
    beta = generate_beta(T, p, R)
    epsilon = np.random.normal(0, sigma, n).reshape(n, 1)
    y = X @ beta.reshape(-1, 1) + epsilon

    return X, y


def simulation3(n=400, p=4000, C=3, sigma=0.5):
    T = 20
    R = 10

    beta = generate_beta(T, p, R)
    epsilon = np.random.normal(0, sigma, n).reshape(n, 1)
    mean = np.zeros(p)
    covariance_matrix = np.full((p, p), 1 / (1 + C * np.linalg.norm(beta, ord=0)))
    for i in range(p):
        covariance_matrix[i, i] = 1

    X = np.random.multivariate_normal(mean, covariance_matrix, size=n)
    y = X @ beta.reshape(-1, 1) + epsilon

    return X, y


if __name__ == "__main__":
    np.random.seed(1234)
    X, y = simulation1(400, 4000, 0.3, 0.5)
    beta_chip = CHIP_SDNR().sdnr_continuation(X, y).beta_opt
    print(beta_chip)
