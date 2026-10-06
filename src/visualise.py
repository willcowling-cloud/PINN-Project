import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _to_numpy(arr):
    """Convert PyTorch Tensor or NumPy array to NumPy array safely."""
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def animate_solution(X, T, predicted_values, exact_values=None, title="PINN Solution",
                      xlabel="x", ylabel="u(x, t)", interval=40, save_path=None):
    """
    Animate a PINN's predicted solution over time, optionally alongside a
    reference solution for comparison.
    """
    # 1. Convert all inputs safely to numpy FIRST
    X_np = _to_numpy(X)
    T_np = _to_numpy(T)
    pred_np = _to_numpy(predicted_values)
    exact_np = _to_numpy(exact_values)

    nx, nt = X_np.shape
    x_vals = X_np[:, 0]
    t_vals = T_np[0, :]

    pred = pred_np.reshape(nx, nt)
    exact = exact_np.reshape(nx, nt) if exact_np is not None else None

    fig, ax = plt.subplots(figsize=(8, 5))

    # Fix y-axis limits up front across all frames
    all_vals = [pred] if exact is None else [pred, exact]
    y_min = min(v.min() for v in all_vals)
    y_max = max(v.max() for v in all_vals)
    y_pad = 0.05 * (y_max - y_min + 1e-8)

    pred_line, = ax.plot([], [], label="PINN", color="tab:blue")
    exact_line = None
    if exact is not None:
        exact_line, = ax.plot([], [], label="Exact", color="tab:orange")

    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)

    def init():
        pred_line.set_data([], [])
        time_text.set_text("")
        if exact_line is not None:
            exact_line.set_data([], [])
            return pred_line, exact_line, time_text
        return pred_line, time_text

    def update(frame):
        pred_line.set_data(x_vals, pred[:, frame])
        ax.set_title(f"{title} (t = {t_vals[frame]:.3f})")
        time_text.set_text(f"t = {t_vals[frame]:.3f}")
        if exact_line is not None:
            exact_line.set_data(x_vals, exact[:, frame])
            return pred_line, exact_line, time_text
        return pred_line, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=nt, init_func=init, interval=interval, blit=False
    )

    if save_path is not None:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow")
        else:
            anim.save(save_path)

    plt.close(fig)  # Prevents duplicate static figure in Jupyter

    return fig, anim
