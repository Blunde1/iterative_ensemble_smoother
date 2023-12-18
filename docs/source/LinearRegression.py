# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# ruff: noqa: E402

# %% [markdown]
# # Linear regression with ESMDA
#
# We solve a linear regression problem using ESMDA.
# First we define the forward model as $g(x) = Ax$,
# then we set up a prior ensemble on the linear
# regression coefficients, so $x \sim \mathcal{N}(0, 1)$.
#
# As shown in the 2013 paper by Emerick et al, when a set of
# inflation weights $\alpha_i$ is chosen so that $\sum_i \alpha_i^{-1} = 1$,
# ESMDA yields the correct posterior mean for the linear-Gaussian case.

# %% [markdown]
# ## Import packages

# %%
import numpy as np
from matplotlib import pyplot as plt

from iterative_ensemble_smoother import ESMDA

# %% [markdown]
# ## Create problem data
#
# Some settings worth experimenting with:
#
# - Decreasing `prior_std=1` will pull the posterior solution toward zero.
# - Increasing `num_ensemble` will increase the quality of the solution.
# - Increasing `num_observations / num_parameters`
#   will increase the quality of the solution.

# %%
num_parameters = 25
num_observations = 100
num_ensemble = 100
prior_std = 1
var_eps = 1.0

# %%
rng = np.random.default_rng(42)

# Create a problem with g(x) = A @ x
A = rng.standard_normal(size=(num_observations, num_parameters))


def g(X):
    """Forward model."""
    return A @ X


# Create observations: obs = g(x) + N(0, 1)
x_true = np.linspace(-1, 1, num=num_parameters)
observation_noise = np.sqrt(var_eps) * rng.standard_normal(size=num_observations)
observations = g(x_true) + observation_noise

# Initial ensemble X ~ N(0, prior_std) and diagonal covariance with ones
X = rng.normal(size=(num_parameters, num_ensemble)) * prior_std

# Covariance matches the noise added to observations above
covariance = var_eps * np.ones(num_observations)

# %% [markdown]
# ## Solve the maximum likelihood problem
#
# We can solve $Ax = b$, where $b$ is the observations,
# for the maximum likelihood estimate.
# Notice that unlike using a Ridge model,
# solving $Ax = b$ directly does not use any prior information.

# %%
x_ml, *_ = np.linalg.lstsq(A, observations, rcond=None)

plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.show()

# %% [markdown]
# ## Solve using ESMDA
#
# We crease an `ESMDA` instance and solve the Guass-linear problem.

# %%
smoother = ESMDA(
    covariance=covariance,
    observations=observations,
    alpha=1,
    seed=1,
)

X_i = np.copy(X)
for i, alpha_i in enumerate(smoother.alpha, 1):
    print(
        f"ESMDA iteration {i}/{smoother.num_assimilations()}"
        + f" with inflation factor alpha_i={alpha_i}"
    )
    X_i = smoother.assimilate(X_i, Y=g(X_i))


X_posterior = np.copy(X_i)

# %% [markdown]
# ## Plot and compare solutions
#
# Compare the true parameters with both the ML estimate
# from linear regression and the posterior means obtained using `ESMDA`.

# %%
plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.scatter(
    np.arange(len(x_true)), np.mean(X_posterior, axis=1), label="Posterior mean"
)
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.show()

# %% [markdown]
# We now include the posterior samples as well.

# %%
plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.scatter(
    np.arange(len(x_true)), np.mean(X_posterior, axis=1), label="Posterior mean"
)

# Loop over every ensemble member and plot it
for j in range(num_ensemble):
    # Jitter along the x-axis a little bit
    x_jitter = np.arange(len(x_true)) + rng.normal(loc=0, scale=0.1, size=len(x_true))

    # Plot this ensemble member
    plt.scatter(
        x_jitter,
        X_posterior[:, j],
        label=("Posterior values" if j == 0 else None),
        color="black",
        alpha=0.2,
        s=5,
        zorder=0,
    )
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.show()

# %% [markdown]
# ## Experimental obsrvation looping
#
# ```python
# for (dj, yj, Sigma_eps_j) in (observations, responses, variance):
#     H = LLS coefficients yj on X[mask,:] # 1xp but using X which could be p2xn for p2<p depending on mask
#     Sigma_yj = H @X @ X.T @ H.T # 1x1
#     Sigma_d = Sigma_yj + Sigma_eps_j # 1x1
#     for i in realizations
#         T_ji = (dj - yji) / Sigma_d # 1x1
#         for X_ki in realization_i_of_parameters:
#             if dj not masked for xk:
#                 X_ki = X_ki + cov_xk_yj*T_ji
# ```

# %%
m = len(observations)
p, n = X.shape

X_ = np.copy(X)
Y = g(X_)

X_centered = X_ - np.mean(X_, axis=1, keepdims=True)

# sampled from standard normal
epsilon = np.sqrt(var_eps) * rng.standard_normal(size=(m,n))

# H: (global) average (linear) gradient (operator)
H = Y @ np.linalg.pinv(X_centered)
Y_ = H @ X_
explained_variance = [1.0 - np.var(Y[j,:] - Y_[j,:]) / np.var(Y[j,:]) for j in range(m)]

#Y_adjusted = np.zeros((m,n))

# Loop over observations
for j in range(m):

    # Re-center parameters
    X_centered = X_ - np.mean(X_, axis=1, keepdims=True)
    
    # Get the relevant observation
    dj = observations[[j]]

    # The prediction from current ensemble
    Y_ = H @ X_
    
    # Get the slice of the (global) average (linear) gradient (operator)
    Hj = H[[j],:]

    # Compute covariance of response
    # Note we could compute: Hj @ X_centered @ X_centered.T @ Hj.T / (n-1)
    # But below should be faster
    var_yj = np.cov(Y_[j,:]).reshape((1,1))

    # Because using model in innovations, add unexplained variance to var_dj
    var_yj = var_yj / explained_variance[j]
    
    # Compute covariance of observation
    var_eps_j = var_eps
    var_dj = var_yj + var_eps_j

    # Loop over realizations
    for i in range(n):
        
        # Get perturbated observation
        dj_perturbed = dj + epsilon[j,i]

        # Compute innovations
        # 1. Using forward model
        #innovations = dj_perturbed - g(X_)[j,i]
        # 2. Using statistical model - leads to inflation of variance
        innovations = dj_perturbed - Y_[j,i]
        # 3. Adjusted forward model
        #innovations = dj_perturbed - (Y[j,i] + Y_adjusted[j,i])
        
        # Compute transition matrix
        T_ji = innovations / var_dj

        # Loop over parameters
        for k in range(p):
            
            # Compute cross covariance among xk and yj
            cov_xk_yj = np.array((X_centered[k,:] @ Y_[j,:].T) / (n-1)).reshape((1,1))
            
            # Update realization i at dimension k
            X_[k,i] += (cov_xk_yj @ T_ji)[0,0]

        # Calculate Y_adjusted
        #Y_adjusted[j,i] += (var_yj @ T_ji)[0,0]

# %%
plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.scatter(
    np.arange(len(x_true)), np.mean(X_, axis=1), label="Posterior mean"
)

# Loop over every ensemble member and plot it
for j in range(num_ensemble):
    # Jitter along the x-axis a little bit
    x_jitter = np.arange(len(x_true)) + rng.normal(loc=0, scale=0.1, size=len(x_true))

    # Plot this ensemble member
    plt.scatter(
        x_jitter,
        X_[:, j],
        label=("Posterior values" if j == 0 else None),
        color="black",
        alpha=0.2,
        s=5,
        zorder=0,
    )
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.show()

# %%
### plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.scatter(
    np.arange(len(x_true)), np.mean(X_, axis=1), label="Posterior mean"
)
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.show()

# %% [markdown]
# The plot below should converge to the straight line y=x when $n\to\infty$

# %%
plt.scatter(np.mean(X_, axis=1), np.mean(X_posterior, axis=1), label="vs")
plt.show()

# %%
