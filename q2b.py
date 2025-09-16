import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Reproducibility
np.random.seed(0)

# ---- Target function (corrected) ----
def target_fn(x):
    # x can be scalar or array
    return 1.2 * np.sin(np.pi * x) - np.cos(2.4 * np.pi * x)

# ---- Data ----
x_train = np.arange(-2.0, 2.0 + 1e-9, 0.05).reshape(-1, 1)   # step 0.05
y_train = target_fn(x_train).ravel()

x_test = np.arange(-2.0, 2.0 + 1e-9, 0.01).reshape(-1, 1)    # step 0.01
y_test = target_fn(x_test).ravel()

# ---- Forward pass for a 1-n-1 MLP (tanh hidden, linear output) ----
def forward(params, x, hidden_units):
    """
    params: 1D array of length 3*hidden_units + 1
      layout: [w1 (n), b1 (n), w2 (n), b2 (1)]
      where w1 is shape (1, n) flattened, b1 is (n,), w2 is (n,1) flattened, b2 is scalar.
    x: shape (m,1)
    returns: predictions shape (m,)
    """
    n = hidden_units
    # unpack
    w1 = params[0:n].reshape(1, n)             # (1, n)
    b1 = params[n:2*n].reshape(1, n)           # (n,) -> (1,n) for broadcasting
    w2 = params[2*n:3*n].reshape(n, 1)         # (n,1)
    b2 = params[3*n]                           # scalar

    hidden = np.tanh(x @ w1 + b1)              # (m, n)
    out = hidden @ w2 + b2                     # (m, 1)
    return out.ravel()

# ---- Residual (cost) function for least_squares ----
def residuals(params, x, y, hidden_units):
    return forward(params, x, hidden_units) - y

# ---- Training loop for different hidden sizes ----
hidden_sizes = list(range(1, 11)) + [20]   # 1..10 and 20 (stop at 20 as required)

results = []
plt.figure(figsize=(12, 10))
for idx, n in enumerate(hidden_sizes, 1):
    # number of parameters = n (w1) + n (b1) + n (w2) + 1 (b2) = 3n + 1
    n_params = 3 * n + 1
    init_params = np.random.randn(n_params) * 0.05

    # Levenberg-Marquardt (lm) requires residuals >= parameters (we ensured that)
    res = least_squares(residuals, init_params, args=(x_train, y_train, n), method='lm',
                        xtol=1e-10, ftol=1e-10, max_nfev=2000)

    # Evaluate
    y_train_pred = forward(res.x, x_train, n)
    y_test_pred  = forward(res.x, x_test, n)
    train_mse = np.mean((y_train_pred - y_train) ** 2)
    test_mse  = np.mean((y_test_pred  - y_test)  ** 2)
    pred_minus3 = forward(res.x, np.array([[-3.0]]), n)[0]
    pred_plus3  = forward(res.x, np.array([[ 3.0]]), n)[0]

    results.append({
        'n_hidden': n,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'pred_-3': pred_minus3,
        'pred_+3': pred_plus3,
        'success': res.success,
        'cost': res.cost
    })

    # Plot (small grid)
    plt.subplot(4, 3, idx)
    plt.plot(x_test.ravel(), y_test, '--', linewidth=1, label='True')
    plt.plot(x_test.ravel(), y_test_pred, '-', linewidth=1, label='MLP')
    plt.scatter(x_train.ravel(), y_train, s=8, alpha=0.6)
    plt.title(f"n= {n}")
    if idx == 1:
        plt.legend(fontsize=8)

plt.tight_layout()
plt.show()

# ---- Print summary ----
print("For x = -3: " + str(target_fn(-3)))
print("For x = 3: " + str(target_fn(3)))
for r in results:
    print(f"n={r['n_hidden']:2d} | "f"f(-3)={r['pred_-3']:.4f} | f(+3)={r['pred_+3']:.4f}")
