"""
Overfitting Study
=========================================================

Comparing training loss and test loss in the training progress to observe overfitting.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append(r'e:\ML_AD_WinterWork\Backpropagation')
from backpropagation_line_search import NeuralNetworkLineSearch


# ============================================================
# Font & random seed/ using chinese when test for the first time
# ============================================================
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

random.seed(42)
np.random.seed(42)



# Normalization function 

def normalize_input(sample):
    """
    Apply min-max normalization to 3 raw fruit features and add 4th "acidity/sweetness ratio" feature.

     Feature ranges:
        直径 diameter  : [90, 120]  → (x - 90) / 30
        甜度 sweetness : [7, 16]   → x / 16
        酸度 acidity   : [4, 10]   → x / 16
        酸甜比 ratio   : acidity / sweetness

    Args:
        sample: [diameter, sweetness, acidity] / raw features

    Returns:
        [diameter_norm, sweetness_norm, acidity_norm, ratio_feature]
    """
    diameter, sweetness, acidity = sample
    diameter_norm  = (diameter - 90) / (120 - 90)
    sweetness_norm = sweetness / 16
    acidity_norm   = acidity / 16
    ratio_feature  = acidity / sweetness
    return [diameter_norm, sweetness_norm, acidity_norm, ratio_feature]


# ============================================================
# Classification rules:
#   label = 1 when all 3 conditions are met:
#     1. diameter >= 105
#     2. sweetness ∈ [11, 15]
#     3. acidity/sweetness ∈ [0.55, 0.65]
#
# 数据生成方式 / Data generation method:
#   遍历 diameter × sweetness × ratio_candidates 的网格,
#   计算 acidity = sweetness × ratio, 保留合法范围, 标注标签。
#   Grid over diameter × sweetness × ratio_candidates,
#   compute acidity = sweetness × ratio, keep valid range, assign label.

print("=" * 35)
print("  Fruit Classification Overfitting Study")
print("=" * 35)

print("\n生成训练数据 / Generating training data ...")

training_data_fruit = []
ratio_candidates = [0.50, 0.54, 0.58, 0.62, 0.66, 0.70]

for diameter in range(90, 121, 2):
    for sweetness in range(7, 17):
        for ratio in ratio_candidates:
            acidity = round(sweetness * ratio, 1)
            if 4 <= acidity <= 10:
                label = 1 if (diameter >= 105
                              and 11 <= sweetness <= 15
                              and 0.55 <= ratio <= 0.65) else 0
                training_data_fruit.append(([diameter, sweetness, acidity], [label]))

# Hard negative mandatory selection + balanced sampling ----
# 目的: 防止模型学到"高直径+高甜度=正例"的捷径, 强制学习规则的细节边界
# To prevent the model from learning "high diameter + high sweetness = positive"

pos_samples = [item for item in training_data_fruit if item[1][0] == 1]
neg_samples = [item for item in training_data_fruit if item[1][0] == 0]

# Select hard negatives: diameter >= 110 but negative due to sweetness or ratio violation
hard_negatives = []
for sample in neg_samples:
    diameter, sweetness, acidity = sample[0]
    ratio = acidity / sweetness
    if diameter >= 110 and (
        sweetness > 15 or ratio < 0.55 or ratio > 0.65
    ):
        hard_negatives.append(sample)

# 去重 (按特征元组) / Deduplicate by feature tuple
seen_features = set()
hard_negatives_unique = []
for sample in hard_negatives:
    key = tuple(sample[0])
    if key not in seen_features:
        seen_features.add(key)
        hard_negatives_unique.append(sample)

# Fill remaining negatives to match positive count
remaining_negatives = [item for item in neg_samples if tuple(item[0]) not in seen_features]
np.random.shuffle(pos_samples)
np.random.shuffle(remaining_negatives)

required_neg_count = len(pos_samples)
selected_negatives = hard_negatives_unique + remaining_negatives
selected_negatives = selected_negatives[:required_neg_count]

training_data_fruit = pos_samples[:len(pos_samples)] + selected_negatives
np.random.shuffle(training_data_fruit)

# Normalizing
training_data_normalized = [(normalize_input(x), y) for x, y in training_data_fruit]

print(f"  正例 / Positive: {len(pos_samples)},  "
      f"负例 / Negative: {len(selected_negatives)},  "
      f"总计 / Total: {len(training_data_fruit)}")
print(f"  困难负例 / Hard negatives: {len(hard_negatives_unique)}")


# ============================================================
# Test Data 
# =========================================================

test_data_fruit_raw = [
    # --- Positives ---
    ([106, 11, 6.1], [1]),
    ([110, 13, 7.5], [1]),
    ([118, 15, 9.2], [1]),
    ([120, 12, 7.1], [1]),

    # --- Negatives ---
    ([104, 12, 7.0], [0]),
    ([102, 14, 8.0], [0]),

    # --- Negatives: only sweetness violated ---
    ([110, 10, 6.0], [0]),
    ([115, 16, 9.4], [0]),

    # --- Negatives: only ratio violated ---
    ([112, 13, 6.4], [0]),   # ratio ≈ 0.49 < 0.55
    ([114, 12, 8.2], [0]),   # ratio ≈ 0.68 > 0.65
    ([116, 11, 5.5], [0]),   # ratio = 0.50 < 0.55
    ([106, 14, 6.8], [0]),   # ratio ≈ 0.49 < 0.55

    # --- Boundary samples ---
    ([105, 11, 6.1], [1]),   # Just inside boundary
    ([105, 11, 6.0], [0]),   # Just outside boundary
]

test_data_fruit = [(normalize_input(x), y) for x, y in test_data_fruit_raw]

print(f" Test samples: {len(test_data_fruit)}")

LAYERSIZE = [4, 14, 14, 14, 14, 1] # a smaller network to make overfitting more likely
print("\nuse residual?")
USE_RESIDUAL = input("  1 for yes, 0 for no (default 0): ") == "1"
#residual 

model = NeuralNetworkLineSearch(LAYERSIZE, use_residual=USE_RESIDUAL)

print(f"\n Architecture: {LAYERSIZE}")
print(f" Residual: {USE_RESIDUAL}")


# ============================================================
# Training Loop
# ============================================================

train_inputs  = [x for x, y in training_data_normalized]
train_targets = [y for x, y in training_data_normalized]
test_inputs   = [x for x, y in test_data_fruit]
test_targets  = [y for x, y in test_data_fruit]

# 训练超参数 / Training hyperparameters
TOTAL_EPOCHS  = 400         #  Total training epochs 
INITIAL_RATE  = 10          #  Line search initial step size
EVAL_INTERVAL = 10          #  Evaluate test set every N epochs

# 记录历史 / History recording
train_loss_history = []    #  Loss / Train loss per epoch
test_loss_history  = []    #  Loss / Test loss per eval
test_eval_epochs   = []    #  Epochs at which test was evaluated

print(f"\n训练参数 / Training config:")
print(f" 总轮数 / Epochs: {TOTAL_EPOCHS}")
print(f"初始步长 / Initial rate: {INITIAL_RATE}")
print(f"测试集评估间隔 / Test eval interval: {EVAL_INTERVAL} epochs")
print(f"开始训练 / Starting training ...")
print("-" * 60)

for epoch in range(TOTAL_EPOCHS):
   
    loss, alpha = model.train_batch_with_line_search(
        train_inputs, train_targets, initial_rate=INITIAL_RATE
    )
    train_loss_history.append(loss)

    # ---- 每 EVAL_INTERVAL 轮评估测试集 n Evaluate test set every EVAL_INTERVAL epochs 
    if (epoch + 1) % EVAL_INTERVAL == 0:
        # Runs forward pass on test set, computes average BCE loss
        test_loss = model.compute_batch_loss(test_inputs, test_targets)
        test_loss_history.append(test_loss)
        test_eval_epochs.append(epoch + 1)

        # 打印进度 / Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:>4}: "
                  f"Train Loss={loss:.6f},  "
                  f"Test Loss={test_loss:.6f},  "
                  f"α={alpha:.6f}")

print(" -------------------Training finished.\n")


# ============================================================
# 过拟合分析 / Overfitting Analysis
# ============================================================
# Find the minimum test loss point
if test_loss_history:
    min_test_loss = min(test_loss_history)
    min_test_epoch = test_eval_epochs[test_loss_history.index(min_test_loss)]
    final_train_loss = train_loss_history[-1]
    final_test_loss = test_loss_history[-1]


# ============================================================
# Plotting

fig, ax = plt.subplots(figsize=(12, 6))

# Train loss: one point per epoch
ax.plot(range(1, TOTAL_EPOCHS + 1), train_loss_history,
        'b-', label='Train Loss', alpha=0.7, linewidth=1)
# Test loss
ax.plot(test_eval_epochs, test_loss_history,
        'r-o', label='Test Loss', alpha=0.8, markersize=3, linewidth=1.5)

# 标注测试 Loss 最低点 / Mark minimum test loss
if test_loss_history:
    _min_val = min(test_loss_history)
    _min_idx = test_loss_history.index(_min_val)
    _min_ep  = test_eval_epochs[_min_idx]
    ax.axvline(x=_min_ep, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate(f'Min Test Loss\n@ Epoch {_min_ep}\n({_min_val:.4f})',
                xy=(_min_ep, _min_val),
                xytext=(_min_ep + TOTAL_EPOCHS * 0.05, _min_val),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
                fontsize=9, color='green')

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss (BCE)', fontsize=11)
ax.set_title(
    f'Fruit Classification: Train vs Test Loss (Overfitting Check)\n'
    f'Architecture: {LAYERSIZE}, Samples: {len(train_inputs)} train / {len(test_inputs)} test',
    fontsize=11
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = 'plot_overfitting_fruit.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {save_path}")
plt.show()  
plt.close(fig)  
