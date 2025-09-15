import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y) -> float:
    return ((1 - x) ** 2) + (100 * (y - x ** 2) ** 2)

def rosenbrock_gradient(x, y):
    dx = (-2 * (1 - x)) - (4 * 100 * x * (y - x**2))
    dy = 2 * 100 * (y - x ** 2)
    return dx, dy

def rosenbrock_gradient_descend(learning_rate=0.001, num_iterations=100000):
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)
    print(f"Initial point: (x, y) = ({x:.6f}, {y:.6f}), f(x,y) = {rosenbrock(x,y):.6e}")

    trajectory = [(x, y)]
    values = [rosenbrock(x, y)]

    tolerance = 1e-6

    for i in range(num_iterations):
        grad_x, grad_y = rosenbrock_gradient(x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        new_x_y = rosenbrock(x, y)

        trajectory.append((x, y))
        values.append(new_x_y)

        if new_x_y < tolerance:
            return x, y, i + 1, trajectory, values
        
    return x, y, num_iterations, trajectory, values

def q1a():
    x_opt, y_opt, iters, traj, vals = rosenbrock_gradient_descend(learning_rate=0.001)
    print(f"Converged to (x, y) = ({x_opt:.6f}, {y_opt:.6f}) in {iters} iterations")
    print(f"Final f(x,y) = {vals[-1]:.6e}")

    # Plot trajectory in 2D
    traj = np.array(traj)
    plt.figure(figsize=(12,5))

    # Trajectory plot
    plt.subplot(1,2,1)
    plt.plot(traj[:,0], traj[:,1], 'o-', markersize=2)
    plt.plot(1, 1, 'r*', markersize=12, label='Global Minimum (1,1)')
    plt.title("Trajectory of (x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # Function value vs iterations
    plt.subplot(1,2,2)
    plt.plot(vals)
    plt.yscale('log')  # log scale to show convergence clearly
    plt.title("Function Value vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("f(x,y)")

    plt.tight_layout()
    plt.show()

    # --- Run with eta = 0.5 (large learning rate) ---
    _, _, _, traj_big, vals_big = rosenbrock_gradient_descend(learning_rate=0.5)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(np.array(traj_big)[:,0], np.array(traj_big)[:,1], 'o-', markersize=2)
    plt.plot(1, 1, 'r*', markersize=12, label='Global Minimum (1,1)')
    plt.title("Trajectory with Large Learning Rate (0.5)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(vals_big)
    plt.yscale('log')
    plt.title("Function Value vs Iteration (eta=0.5)")
    plt.xlabel("Iteration")
    plt.ylabel("f(x,y)")

    plt.tight_layout()
    plt.show()

q1a()


    


