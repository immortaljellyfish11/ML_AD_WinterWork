"""
Backtracking Line Search Computational Cost Analysis
See line_search_cost_analysis_notes.md for detailed documentation.
"""

import time
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

sys.path.append(r'e:\ML_AD_WinterWork\Backpropagation')
from backpropagation_line_search import NeuralNetworkLineSearch

# Font configuration 
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# Forward Pass Counter — monkey-patches model.forward() to count calls
# ============================================================

class ForwardCounter:
    """Wraps model.forward to count invocations without modifying the class file."""

    def __init__(self, model):
        self.model = model
        self.count = 0
        self._original_forward = model.forward
        model.forward = self._counting_forward

    def _counting_forward(self, *args, **kwargs):
        self.count += 1
        return self._original_forward(*args, **kwargs)

    def reset(self):
        self.count = 0

    def restore(self):
        self.model.forward = self._original_forward


# ============================================================
# Fixed-LR Batch Gradient Descent (external function, reuses existing gradient logic)
# ============================================================

def train_fixed_lr_batch(model, inputs, targets, learning_rate):
    """
    Train one epoch using BGD with a fixed learning rate.
    Cost: 2n forward passes per epoch (n for gradients + n for post-update loss).
    """
    # Compute full-batch average gradients (n forward passes internally)
    grad_weights, grad_biases = model.compute_batch_gradients(inputs, targets)

    # Update weights with fixed step size (no Armijo check)
    for l in range(model.layer_num - 1):
        for i in range(model.layersize[l + 1]):
            model.biases[l][i] -= learning_rate * grad_biases[l][i]
            for k in range(model.layersize[l]):
                model.weights[l][i][k] -= learning_rate * grad_weights[l][i][k]

    # Compute post-update batch loss (n forward passes internally)
    loss = model.compute_batch_loss(inputs, targets)
    return loss


# ============================================================
# Dataset Generation
# ============================================================

def generate_xor_data():
    inputs  = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0],    [1],    [1],    [0]]
    return inputs, targets


def generate_polynomial_data(n_samples=100):
    """
    Polynomial classification dataset.
    Label: sin(x1) + x2^3 - x3^2 + 0.5*x1*x3 > 0 -> 1, else 0
    Features normalized from [-10, 10] to [0, 1].
    """
    X = np.random.uniform(-10, 10, (n_samples, 3))
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

    y_raw = np.sin(x1) + x2**3 - x3**2 + 0.5 * x1 * x3
    y = (y_raw > 0).astype(int)

    inputs  = [[(x + 10) / 20 for x in row] for row in X.tolist()]
    targets = [[int(y[i])] for i in range(len(y))]
    return inputs, targets


# ============================================================
# Core Experiment Function
# ============================================================

def run_experiment(dataset_name, inputs, targets, layersize,
                   fixed_lr, initial_alpha, max_epochs, target_losses):
    """
    Run "Fixed LR vs Line Search" comparison on one dataset.
    Both networks start from identical weights. Records per-epoch loss,
    cumulative forward count, and wall-clock time. Detects when each
    target loss threshold is first reached.
    """
    n_samples = len(inputs)
    print(f"\n{'=' * 65}")
    print(f"  Experiment: {dataset_name}")
    print(f"  Samples: {n_samples},  Arch: {layersize}")
    print(f"  Fixed LR: {fixed_lr},  LS init alpha: {initial_alpha}")
    print(f"  Max Epochs: {max_epochs},  Targets: {target_losses}")
    print(f"{'=' * 65}")

    # Initialize two networks with identical weights
    random.seed(42)
    np.random.seed(42)

    model_fixed = NeuralNetworkLineSearch(layersize, use_residual=False)

    model_ls = NeuralNetworkLineSearch(layersize, use_residual=False)
    model_ls.weights = copy.deepcopy(model_fixed.weights)
    model_ls.biases  = copy.deepcopy(model_fixed.biases)

    # Install forward pass counters
    counter_fixed = ForwardCounter(model_fixed)
    counter_ls    = ForwardCounter(model_ls)

    results = {
        'fixed':       {'losses': [], 'forward_counts': [], 'times': [], 'thresholds': {}},
        'line_search': {'losses': [], 'forward_counts': [], 'times': [], 'thresholds': {}},
    }

    # ---- Train Method A: Fixed LR BGD ----
    print(f"\n  [Method A] Fixed LR BGD (lr={fixed_lr})")
    remaining_a = sorted(target_losses, reverse=True)
    t0 = time.time()

    for epoch in range(max_epochs):
        loss = train_fixed_lr_batch(model_fixed, inputs, targets, fixed_lr)
        elapsed = time.time() - t0

        if math.isnan(loss) or math.isinf(loss):
            print(f"    [!] Loss diverged at epoch {epoch + 1}. Stopping.")
            break

        results['fixed']['losses'].append(loss)
        results['fixed']['forward_counts'].append(counter_fixed.count)
        results['fixed']['times'].append(elapsed)

        for tl in remaining_a[:]:
            if loss <= tl:
                results['fixed']['thresholds'][tl] = {
                    'epoch': epoch + 1,
                    'forward_count': counter_fixed.count,
                    'time': elapsed,
                    'loss': loss
                }
                remaining_a.remove(tl)
                print(f"    ✓ Loss ≤ {tl} @ Epoch {epoch + 1}, "
                      f"Forwards={counter_fixed.count}, Time={elapsed:.3f}s")

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch + 1:>5}: Loss={loss:.6f}, "
                  f"Forwards={counter_fixed.count}")

        if not remaining_a:
            print(f"    All targets reached @ epoch {epoch + 1}")
            break

    if remaining_a:
        print(f"    Unreached targets: {remaining_a}")

    # ---- Train Method B: Armijo Line Search BGD ----
    print(f"\n  [Method B] Armijo Line Search BGD (init_alpha={initial_alpha})")
    remaining_b = sorted(target_losses, reverse=True)
    t0 = time.time()

    for epoch in range(max_epochs):
        loss, alpha = model_ls.train_batch_with_line_search(
            inputs, targets, initial_rate=initial_alpha
        )
        elapsed = time.time() - t0

        if math.isnan(loss) or math.isinf(loss):
            print(f"    [!] Loss diverged at epoch {epoch + 1}. Stopping.")
            break

        results['line_search']['losses'].append(loss)
        results['line_search']['forward_counts'].append(counter_ls.count)
        results['line_search']['times'].append(elapsed)

        for tl in remaining_b[:]:
            if loss <= tl:
                results['line_search']['thresholds'][tl] = {
                    'epoch': epoch + 1,
                    'forward_count': counter_ls.count,
                    'time': elapsed,
                    'loss': loss
                }
                remaining_b.remove(tl)
                print(f"    ✓ Loss ≤ {tl} @ Epoch {epoch + 1}, "
                      f"Forwards={counter_ls.count}, Time={elapsed:.3f}s, "
                      f"α={alpha:.6f}")

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch + 1:>5}: Loss={loss:.6f}, "
                  f"Forwards={counter_ls.count}, α={alpha:.6f}")

        if not remaining_b:
            print(f"    All targets reached @ epoch {epoch + 1}")
            break

    if remaining_b:
        print(f"    Unreached targets: {remaining_b}")

    counter_fixed.restore()
    counter_ls.restore()

    return results


# ============================================================
# 绘图函数 / Plotting Function
# ============================================================

def plot_comparison(results, dataset_name, target_losses, save_prefix):
    """
    绘制三合一对比图。
    Draw a three-panel comparison figure.
          Plot 1 (Loss vs Forward Count): 
          Plot 2 (Loss vs Wall Time)
          Plot 3 (Bar chart)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 颜色与标签 / Colors and labels
    colors = {'fixed': '#2196F3', 'line_search': '#F44336'}
    labels = {
        'fixed':       'Fixed LR (BGD) ',
        'line_search': 'Armijo Line Search (BGD) '
    }

    # ────── 图1: Loss vs Forward Pass Count ──────
    ax = axes[0]
    for method in ['fixed', 'line_search']:
        d = results[method]
        if d['losses']:
            ax.plot(d['forward_counts'], d['losses'],
                    color=colors[method], label=labels[method],
                    linewidth=1.5, alpha=0.85)

    # Draw target threshold horizontal lines
    for tl in target_losses:
        ax.axhline(y=tl, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Cumulative Forward Passes', fontsize=10)
    ax.set_ylabel('Loss (BCE)', fontsize=10)
    ax.set_title(f'{dataset_name}\nLoss vs. Computational Cost (Forward Passes)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Loss vs Wall-Clock Time
    ax = axes[1]
    for method in ['fixed', 'line_search']:
        d = results[method]
        if d['losses']:
            ax.plot(d['times'], d['losses'],
                    color=colors[method], label=labels[method],
                    linewidth=1.5, alpha=0.85)

    for tl in target_losses:
        ax.axhline(y=tl, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Wall-Clock Time (s)', fontsize=10)
    ax.set_ylabel('Loss (BCE)', fontsize=10)
    ax.set_title(f'{dataset_name}\nLoss vs. Wall-Clock Time', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Bar chart — forward passes to reach each target
    ax = axes[2]

    # Collect data for each threshold
    bar_labels = []
    fixed_counts = []
    ls_counts = []

    for tl in sorted(target_losses, reverse=True):
        f_data  = results['fixed']['thresholds'].get(tl)
        ls_data = results['line_search']['thresholds'].get(tl)

        # Show only if at least one method reached it
        if f_data or ls_data:
            bar_labels.append(f'Loss≤{tl}')
            fixed_counts.append(f_data['forward_count'] if f_data else 0)
            ls_counts.append(ls_data['forward_count'] if ls_data else 0)

    if bar_labels:
        x = np.arange(len(bar_labels))
        width = 0.35

        bars_f  = ax.bar(x - width / 2, fixed_counts, width,
                         label='Fixed LR', color=colors['fixed'], alpha=0.8)
        bars_ls = ax.bar(x + width / 2, ls_counts, width,
                         label='Line Search', color=colors['line_search'], alpha=0.8)

        # Annotate values above bars
        for bar in bars_f:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{int(h)}',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=8)
            else:
                ax.annotate('N/R',
                            xy=(bar.get_x() + bar.get_width() / 2, 0),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=7, color='gray')

        for bar in bars_ls:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{int(h)}',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=8)
            else:
                ax.annotate('N/R',
                            xy=(bar.get_x() + bar.get_width() / 2, 0),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=7, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No target reached',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)

    ax.set_xlabel('Target Accuracy', fontsize=10)
    ax.set_ylabel('Forward Passes', fontsize=10)
    ax.set_title(f'{dataset_name}\nCost to Reach Target Accuracy',
                 fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = f'{save_prefix}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {save_path}")
    # plt.show()  # Uncomment for interactive viewing
    plt.close(fig)


def print_summary_table(results, target_losses, dataset_name):
    """Print a text-formatted comparison summary table with speedup ratios."""
    print(f"\n{'=' * 75}")
    print(f"  Comparison Summary -- {dataset_name}")
    print(f"{'=' * 75}")
    print(f"  {'Target':<10} {'Method':<24} {'Epoch':<8} "
          f"{'Forwards':<12} {'Time(s)':<10} {'Speedup':<10}")
    print(f"  {'-' * 70}")

    for tl in sorted(target_losses, reverse=True):
        f_data  = results['fixed']['thresholds'].get(tl)
        ls_data = results['line_search']['thresholds'].get(tl)

        # Fixed LR row
        if f_data:
            print(f"  ≤{tl:<8.3f} {'Fixed LR':<24} {f_data['epoch']:<8} "
                  f"{f_data['forward_count']:<12} {f_data['time']:<10.4f}", end='')
        else:
            print(f"  ≤{tl:<8.3f} {'Fixed LR':<24} {'N/A':<8} "
                  f"{'N/A':<12} {'N/A':<10}", end='')
        print()  # no speedup for first row

        # Line search row
        if ls_data:
            # Speedup ratio: fixed_forwards / ls_forwards
            speedup_str = ''
            if f_data and ls_data:
                ratio = f_data['forward_count'] / ls_data['forward_count']
                if ratio > 1:
                    speedup_str = f'LS {ratio:.1f}x faster'
                elif ratio < 1:
                    speedup_str = f'Fixed {1/ratio:.1f}x faster'
                else:
                    speedup_str = 'same'
            print(f"  {'':10} {'Armijo Line Search':<24} {ls_data['epoch']:<8} "
                  f"{ls_data['forward_count']:<12} {ls_data['time']:<10.4f} "
                  f"{speedup_str}")
        else:
            print(f"  {'':10} {'Armijo Line Search':<24} {'N/A':<8} "
                  f"{'N/A':<12} {'N/A':<10}")

        print(f"  {'-' * 70}")


# Main
if __name__ == "__main__":
    print("=" * 65)
    print("  Backtracking Line Search Computational Cost Analysis")
    print("=" * 65)

    # Experiment 1: XOR (4 samples)
    print("\n\n" + "-" * 65)
    print("  Experiment 1: XOR")
    print("-" * 65)

    xor_inputs, xor_targets = generate_xor_data()
    results_xor = run_experiment(
        dataset_name="XOR Problem",
        inputs=xor_inputs,
        targets=xor_targets,
        layersize=[2, 10, 10, 10, 1],
        fixed_lr=0.4,
        initial_alpha=5.0,
        max_epochs=500,
        target_losses=[0.5, 0.3, 0.1, 0.05]
    )
    print_summary_table(results_xor, [0.5, 0.3, 0.1, 0.05], "XOR Problem")
    plot_comparison(results_xor, "XOR Problem", [0.5, 0.3, 0.1, 0.05],
                    save_prefix="plot_cost_xor")

    # Experiment 2: Polynomial Classification (100 samples)
    print("\n\n" + "-" * 65)
    print("  Experiment 2: Polynomial Classification")
    print("-" * 65)

    np.random.seed(123)  # separate seed from XOR experiment
    poly_inputs, poly_targets = generate_polynomial_data(n_samples=100)
    results_poly = run_experiment(
        dataset_name="Polynomial Classification",
        inputs=poly_inputs,
        targets=poly_targets,
        layersize=[3, 15, 15, 15, 15, 15, 1],
        fixed_lr=0.4,
        initial_alpha=5.0,
        max_epochs=400,
        target_losses=[0.65, 0.55, 0.45, 0.35]
    )
    print_summary_table(results_poly, [0.65, 0.55, 0.45, 0.35],
                        "Polynomial Classification")
    plot_comparison(results_poly, "Polynomial Classification",
                    [0.65, 0.55, 0.45, 0.35],
                    save_prefix="plot_cost_polynomial")

    # Done
    print("\n" + "=" * 65)
    print("  All experiments finished.")
    print("  Output: plot_cost_xor.png, plot_cost_polynomial.png")
    print("=" * 65)
