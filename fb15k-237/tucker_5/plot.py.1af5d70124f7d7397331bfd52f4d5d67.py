import numpy as np
import tensorly as tl
import lingam
import graphviz
from lingam.utils import make_dot, print_causal_directions
from scipy.io import savemat
from sklearn.decomposition import PCA

e = np.loadtxt(fname="ent_embedding.tsv", delimiter="\t", skiprows=0)
w = np.loadtxt(fname="W.tsv", delimiter="\t", skiprows=0)
r = np.loadtxt(fname="rel_embedding.tsv", delimiter="\t", skiprows=0)

x = w.shape[0] if w.shape[0] < w.shape[1] else w.shape[1]

w = w.reshape(x, x, x)

n_r = 237

Q_tensor = []
m1 = tl.tenalg.mode_dot(w, e, 1, transpose=False)
m_new = tl.tenalg.mode_dot(w, r, 1, transpose=False)

print(f"the shape of m_new{m_new.shape}")

m_new = m_new.reshape(m_new.shape[1], m_new.shape[0] * m_new.shape[2])

print(f"new shape : {m_new.shape}")


for i in range(n_r):
    m2 = tl.tenalg.mode_dot(m1, r[i], 2, transpose=False)
    print(m2.shape)
    m3 = np.dot(m2, e)

    Q_tensor.append(m3)
Q_tensor = np.array(Q_tensor)
Q_matrix = Q_tensor.reshape(Q_tensor.shape[0], Q_tensor.shape[1] * Q_tensor.shape[2])

print(Q_matrix.shape)
#pca = PCA(n_components=10, svd_solver="arpack")
#Q_matrix = pca.fit(Q_matrix).transform(Q_matrix)


# noise = np.random.rand(2360, 10)


# Q_matrix = Q_matrix + noise

# rho = np.corrcoef(Q_matrix)
# print(".................................................................")

# print(rho)


# print("...................................................................")
# with open("q_matrix.npy", "wb") as f:
# np.save(f, Q_matrix)

# f.close()

# mdic = {"Q": Q_matrix}
# savemat("Q_matrix.mat", mdic)

"""


def pwling(Q_matrix):
    X = Q_matrix
    n, T = Q_matrix.shape
    bleh = np.mean(X.T, axis=0).T
    print("============================================")
    print(bleh)
    print("=================================================")
    print(bleh.shape)
    bleh = bleh.reshape(bleh.shape[0], 1)

    X = X - np.dot(bleh, np.ones((1, T)))
    bleh_std = np.std(X.T, axis=0) + np.finfo(float).eps
    print("==================================================")
    print(bleh_std.shape)
    print("==================================================")

    bleh_std = bleh_std.reshape(bleh_std.shape[0], 1)

    X = np.divide(X, np.dot(bleh_std, np.ones((1, T))))
    C = np.cov(X.T)
    LR = np.zeros((n, n))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print(LR.shape)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i in range(n):
        for j in range(n):
            if i != j:
                res1 = X[j, :] - np.dot(C[j, i], X[i, :])
                res2 = X[i, :] - np.dot(C[i, j], X[j, :])
                LR[i, j] = (
                    mentaprr(X[j, :])
                    - mentaprr(X[i, :])
                    - mentaprr(res1)
                    + mentaprr(res2)
                )
    return LR


def mentaprr(x):

    x = x - np.mean(x)
    xstd = np.std(x)
    x = x / xstd

    k1 = 36 / (8 * np.sqrt(3) - 9)
    gamma = 0.37457
    k2 = 79.047
    gaussianEntropy = np.log(2 * np.pi) / 2 + 1 / 2

    negentropy = k2 * pow((np.mean(np.log(np.cosh(x))) - gamma), 2) + k1 * pow(
        (np.mean(np.multiply(x, np.exp(-np.power(x, 2) / 2)))), 2
    )

    entropy = gaussianEntropy - negentropy + np.log(xstd)

    return entropy


# test_matrix = pwling(Q_matrix.T)
# print(test_matrix)

"""

model = lingam.DirectLiNGAM()


# model2 = lingam.DirectLiNGAM()


# result = model.bootstrap(m_new, n_sampling=100)
# cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.01, split_by_causal_effect_sign=True)
# print_causal_directions(cdc, 100)

print(Q_matrix.shape)
model.fit(Q_matrix)

print(model.adjacency_matrix_)

# L = np.random.uniform(5, 5)
# model2.fit(L)

print("DONE FITTING THE PROBLEM")
p_values = model.get_error_independence_p_values(Q_matrix)
print(p_values)

print(f"The mean of the p-values is: {np.mean(p_values)}")

print(f"The median of the p-values is:{np.median(p_values)}")


print("DONE GETTING THE P-VALUES")
# ent = model._entropy()


dot = make_dot(model.adjacency_matrix_)
dot.format = "png"
dot.render("dag1")
