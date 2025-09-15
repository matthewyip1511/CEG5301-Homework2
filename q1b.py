import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock function
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Gradient
def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 4 * 100 * x * (y - x ** 2)
    dy = 2 * 100 * (y - x ** 2)
    return np.array([dx, dy])

# Hessian
def rosenbrock_hessian(x, y, b=100):
    h11 = 2 - 4*b*(y - x**2) + 8*b*x**2
    h12 = -4*b*x
    h21 = -4*b*x
    h22 = 2*b
    return np.array([[h11, h12],
                     [h21, h22]])

def newton_method(max_iter=100, tol=1e-6):
    # Random initialization in (0,1)
    x, y = np.random.uniform(0, 1), np.random.uniform(0, 1)

    trajectory = [(x, y)]
    values = [rosenbrock(x, y)]

    for i in range(max_iter):
        grad = rosenbrock_grad(x, y)
        hess = rosenbrock_hessian(x, y)

        # Solve H * step = grad  → step = H^-1 grad
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Hessian not invertible, stopping.")
            break

        x, y = np.array([x, y]) - step
        f_val = rosenbrock(x, y)

        trajectory.append((x, y))
        values.append(f_val)

        if f_val < tol:
            return x, y, i+1, trajectory, values

    return x, y, max_iter, trajectory, values

# --- Run Newton's Method ---
x_opt, y_opt, iters, traj, vals = newton_method()
print(f"Converged to (x, y) = ({x_opt:.6f}, {y_opt:.6f}) in {iters} iterations")
print(f"Final f(x,y) = {vals[-1]:.6e}")

# Plot trajectory in 2D
traj = np.array(traj)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(traj[:,0], traj[:,1], 'o-', markersize=3)
plt.plot(1, 1, 'r*', markersize=12, label='Global Minimum (1,1)')
plt.title("Trajectory of (x,y) with Newton's Method")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Function value vs iterations
plt.subplot(1,2,2)
plt.plot(vals, marker='o')
plt.yscale('log')
plt.title("Function Value vs Iteration (Newton's Method)")
plt.xlabel("Iteration")
plt.ylabel("f(x,y)")

plt.tight_layout()
plt.show()
