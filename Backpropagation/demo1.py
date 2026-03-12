from backpropagation_Origin import NeuralNetwork
from backpropagation_line_search import NeuralNetworkLineSearch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 设置中文字体，避免 "Glyph missing from DejaVu Sans" 警告
# Set Chinese font to prevent CJK glyph-missing warnings
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

np.random.seed(42)

def generate_polynomial_dataset(n_samples=1500):
    """
    Generate a polynomial classification dataset to test deep networks against vanishing gradients.

    特征 / Features : 3 维，均匀分布在 [-10, 10] / 3-dim, uniform in [-10, 10]
    标签 / Label    : 二分类 (0/1)，由高度非线性公式生成 / Binary (0/1) from a highly nonlinear formula
    """
    X = np.random.uniform(-10, 10, (n_samples, 3))

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]

    # Highly nonlinear combination (includes sin, cubic term, and cross term)
    y_raw = np.sin(x1) + x2**3 - x3**2 + 0.5*x1*x3

    # Threshold at 0 for binary classification: y_raw > 0 → 1, else → 0
    y = (y_raw > 0).astype(int)

    # Ensure strictly binary int labels (redundant safety check)
    y = (y > 0.5).astype(int)

    return X, y.reshape(-1, 1)

def normalize_input(sample):
    """
    Apply min-max normalization to the three raw fruit features, scaling to [0, 1].

    Feature ranges:
        直径 diameter  : [90, 120]
        甜度 sweetness : [7,  16]
        酸度 acidity   : [4,  10]
        酸甜比 ratio   : acidity / sweetness
    """
    diameter, sweetness, acidity = sample

    # Scale each feature to [0, 1] using (x - x_min) / (x_max - x_min)
    diameter_norm  = (diameter  -  90) / (120 -  90)
    sweetness_norm = sweetness/16
    acidity_norm   = acidity/16

    ratio_feature = acidity / sweetness

    return [diameter_norm, sweetness_norm, acidity_norm, ratio_feature]

def normalize_polynomial_input(sample):
    """
    Normalize polynomial dataset features from [-10, 10] to [0, 1] via min-max scaling.

    Formula: (x - (-10)) / (10 - (-10))  =  (x + 10) / 20
    """
    # 逐元素归一化，x_min = -10, x_max = 10 / Element-wise normalization, x_min=-10, x_max=10
    return [(x - (-10)) / (10 - (-10)) for x in sample]

if __name__ == "__main__":
    SAVE_PATHS = {
        1: "E:/ML_AD_WinterWork/Backpropagation/model_linesearch_xor.pkl",    
        2: "E:/ML_AD_WinterWork/Backpropagation/model_linesearch_fruit.pkl",  
        3: "E:/ML_AD_WinterWork/Backpropagation/model_linesearch_poly.pkl",    
    }
    random_seed = 42
    np.random.seed(random_seed)
    LAYERSIZES = {
        1: [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1],  
        2: [4, 14, 14, 14, 14, 1],  
        3: [3, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1],
    }

    # Prob.1:The XOR problem --training data
    training_data_XOR = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    # Prob.2: 二分类任务 -- training data（规则驱动 + 均衡采样）
    # Rule:
    #   label=1 iff diameter>=105, sweetness in [11,15], acidity/sweetness in [0.55,0.65]
    training_data_fruit = []
    ratio_candidates = [0.50, 0.54, 0.58, 0.62, 0.66, 0.70]

    for diameter in range(90, 121, 2):
        for sweetness in range(7, 17):
            for ratio in ratio_candidates:
                acidity = round(sweetness * ratio, 1)
                if 4 <= acidity <= 10:
                    label = 1 if (diameter >= 105 and 11 <= sweetness <= 15 and 0.55 <= ratio <= 0.65) else 0
                    training_data_fruit.append(([diameter, sweetness, acidity], [label]))

    # 平衡正负样本，避免训练时被多数类主导
    # 同时加入“困难负例必选”：高直径/高甜度但因规则细节应判负（针对 8-10 类错判）
    pos_samples = [item for item in training_data_fruit if item[1][0] == 1]
    neg_samples = [item for item in training_data_fruit if item[1][0] == 0]

    hard_negatives = []
    for sample in neg_samples:
        diameter, sweetness, acidity = sample[0]
        ratio = acidity / sweetness
        if diameter >= 110 and (
            sweetness > 15 or ratio < 0.55 or ratio > 0.65
        ):
            hard_negatives.append(sample)

    # 去重（按特征）防止困难负例重复进入
    seen_features = set()
    hard_negatives_unique = []
    for sample in hard_negatives:
        key = tuple(sample[0])
        if key not in seen_features:
            seen_features.add(key)
            hard_negatives_unique.append(sample)

    # 其余负例随机补齐到与正例同量
    remaining_negatives = [item for item in neg_samples if tuple(item[0]) not in seen_features]
    np.random.shuffle(pos_samples)
    np.random.shuffle(remaining_negatives)
    required_neg_count = len(pos_samples)
    selected_negatives = hard_negatives_unique + remaining_negatives
    selected_negatives = selected_negatives[:required_neg_count]

    training_data_fruit = pos_samples[:len(pos_samples)] + selected_negatives
    np.random.shuffle(training_data_fruit)
    
    # 归一化水果训练数据 / Normalize fruit training data using per-feature min-max scaling
    training_data_fruit_normalized = [(normalize_input(x), y) for x, y in training_data_fruit]

    # Prob.3: Polynomial classification — training data (1500 randomly generated samples)
    X_vanish, Y_vanish = generate_polynomial_dataset()
    # Normalize raw features to [0,1] and convert numpy labels to Python int
    train_data_for_vanish = [
        (normalize_polynomial_input(X_vanish[i].tolist()), [int(Y_vanish[i][0])])
        for i in range(len(X_vanish))
    ]

    print(f"Generated samples for the polynomial problem (Prob.3):")
    print(train_data_for_vanish[:5])   # 打印前 5 条样本预览 / Print first 5 sample previews
    print("\n")

    training_data_map = {
        1: training_data_XOR,
        2: training_data_fruit_normalized,
        3: train_data_for_vanish,
    }

    # 推理测试数据（在此处填写你想推理的样本，程序会自动处理归一化）
    # test data 
    # normalization is handled automatically by the program)

    # --- Prob.1  XOR inference test samples ---
    test_data_XOR = [
        ([0, 0], [0]),   # 预期输出 0 / Expected output: 0
        ([0, 1], [1]),   
        ([1, 0], [1]),   
        ([1, 1], [0]),   
    ]

    #     Prob.2 inference data: fruit (raw feature scale, auto-normalized)
    # Feature format: [直径 diameter (90-120), 甜度 sweetness (7-16), 酸度 acidity (4-10)]
    test_data_fruit_raw = [
        # --- 正例（满足三条规则）---
        ([106, 11, 6.1], [1]),
        ([110, 13, 7.5], [1]),
        ([118, 15, 9.2], [1]),
        ([120, 12, 7.1], [1]),

        # --- 负例：仅违反直径阈值 ---
        ([104, 12, 7.0], [0]),
        ([102, 14, 8.0], [0]),

        # --- 负例：仅违反甜度范围 ---
        ([110, 10, 6.0], [0]),
        ([115, 16, 9.4], [0]),

        # --- 负例：仅违反酸甜比范围 ---
        ([112, 13, 6.4], [0]),  # ratio < 0.55
        ([114, 12, 8.2], [0]),  # ratio > 0.65
        ([116, 11, 5.5], [0]),  # ratio < 0.55
        ([106, 14, 6.8], [0]),

        # --- 边界样本（帮助验证阈值是否学到）---
        ([105, 11, 6.1], [1]),  # 全在边界内
        ([105, 11, 6.0], [0])   # ratio 略低于下界
    ]
    test_data_fruit = [(normalize_input(x), y) for x, y in test_data_fruit_raw]

    #     Prob.3 inference data: polynomial (raw scale [-10, 10], auto-normalized)
    #     特征格式 / Feature format: [x1, x2, x3]，每个分量范围 / each in [-10, 10]
    test_data_poly_raw = [
        ([ 2.5,  3.0, -1.5], [1]),   # 1 / Label unknown — score only
        ([-5.0,  2.0,  4.0], [0]),   # 0
        ([ 0.0, -3.0,  1.0], [0]),   # 0
        ([ 4.0,  0.0, -2.0], [0]),   # 0
        ([ 9.0,  6.5, -0.8], [1]),   # 1
        ([-6.0, -8.4,  9.8], [0]),   # 0
        ([-6.0,  2.4, -8.8], [0]),   # 0
        ([-6.0,  8.4, -8.8], [1]),   # 1
        ([ 8.0, -2.0,  3.0], [0]),   # 0
        ([ 7.0, -4.0, -5.0], [0])    # 0
    ]
  
    test_data_poly = [(normalize_polynomial_input(x), y) for x, y in test_data_poly_raw]

    test_data_map = {1: test_data_XOR, 2: test_data_fruit, 3: test_data_poly}

    # 问题名称（用于打印标题）/ Problem names for display
    problem_names = {
        1: "XOR Problem",
        2: "Fruit Classification",
        3: "Polynomial Classification",
    }

    # ===================================================================
    # Helper function: run forward-pass inference sample-by-sample and print results
    # ======================================================
    def run_inference(model, test_data, problem_name):
        """
        对给定测试数据逐条推理并格式化打印预测结果。
        Run inference on each sample in test_data and print formatted results.

        参数 / Parameters:
            model        : 已加载或训练完毕的 NeuralNetworkLineSearch 实例
                           A loaded or freshly trained NeuralNetworkLineSearch instance
            test_data    : [(inputs, label), ...] 列表，label 可为 None
                           List of (inputs, label) tuples; label may be None
            problem_name : 用于打印标题的问题名称字符串
                           Problem name string printed as the section title
        """
        print(f"\n{'='*55}")
        print(f"[推理结果 / Inference Results]  {problem_name}")
        print(f"{'='*55}")

        for i, (inputs, label) in enumerate(test_data):
            # 前向传播，获取输出层激活值 / Forward pass → output layer activation list
            output = model.reasoning(inputs)
            # 阈值 0.5 做二分类决策 / Binary decision: threshold at 0.5
            pred = 1 if output[0] >= 0.5 else 0
            label_str = str(label[0]) if (label is not None) else "未知 / Unknown"
            print(f"  样本 {i+1:>2} / Sample {i+1:>2}:")
            print(f"    输入  / Input : {inputs}")
            print(f"    得分  / Score : {output[0]:.6f}")
            print(f"    预测  / Pred  : {pred}    |  标签 / Label: {label_str}")

    # =====================================================================
    # Main entry point: ask whether to run inference or train from scratch
    # ===================================================================
    print("\n" + "=" * 55)
    print("  神经网络演示程序 / Neural Network Demo")
    print("=" * 55)
    mode = input(
        "Use a pre-trained model for inference only? "
        "(y=推理/inference, n=训练/train): "
    ).strip().lower()

    # -----------
    # Branch A: Inference Mode
    if mode == 'y':
        print("\n--- Inference Mode ---")

        print("\n请选择要推理的问题 / Select the problem to infer:")
        print("  1 -  XOR Problem")
        print("  2 -  Fruit Classification")
        print("  3 -  Polynomial Classification")
        test_number = int(input("请输入编号 / Enter number (1/2/3): ").strip())

        save_path = SAVE_PATHS[test_number]

        # Check if the problem-specific model file exists
        if not os.path.exists(save_path):
            print(f"\n[Error] problem {test_number} : '{save_path}' do not exist.")
            print(f"  Model file not found. Please train problem {test_number} first (choose 'n').")
            exit(1)

        #  Load trained weights and biases for the selected problem
        print(f"\nLoading '{save_path}' ...")
        model = NeuralNetworkLineSearch.load(save_path)
        print("Model loaded successfully.")

        #  Run inference and print formatted results
        run_inference(model, test_data_map[test_number], problem_names[test_number])

    # ------------------------------------------------------------------
    # Branch B: Training Mode 
    else:
        print("\n--- Training Mode ---")

        # 选择训练问题 / Select problem to train
        print("\n请选择要训练的问题 / Select the problem to train:")
        print("  1 - XOR Problem")
        print("  2 -  Fruit Classification")
        print("  3 -  Polynomial Classification")
        test_number = int(input("Enter number (1/2/3): ").strip())
        print("\nuse residual？")
        use_residual = input(" y/n: ").strip().lower() == 'y'
        train_times = int(input(" Enter number of training epochs: ").strip())

        layersize     = LAYERSIZES[test_number]         # 网络层宽配置 / Layer width config
        save_path     = SAVE_PATHS[test_number]         # 模型保存路径 / Model save path
        training_data = training_data_map[test_number]  # 对应训练集 / Corresponding training set

        # Initialize line-search network (random Xavier weight initialization)
        print(f"\n正在初始化模型 / Initializing model ...")
        print(f"  网络结构 / Layer sizes : {layersize}")
        model = NeuralNetworkLineSearch(layersize=layersize, use_residual=use_residual)

        print(f"\n开始训练 / Starting training — Problem {test_number}: {problem_names[test_number]}")
        print(f"  训练轮数 / Total epochs : {train_times}")
        print("-" * 55)

        loss_plot = []   #  Record lossfor plotting
        for j in range(train_times):
            loss, alpha = model.train_batch_with_line_search(
                [data[0] for data in training_data],    
                [data[1] for data in training_data],    
                initial_rate=10                           
            )
            # Sample loss every 2 epochs
            if j % 2 == 0:
                loss_plot.append(loss)
            #  Print training progress every ** epochs
            if j % 50 == 0:
                print(f"  Epoch {j:>6}: Loss = {loss:.6f},  Alpha = {alpha:.8f}")

        # Training complete 
        model.save(save_path)
        print(f"\n[保存成功 / Saved] 模型已保存至 '{save_path}'")

        # Auto-inference: immediately infer on test data right after training
        # ---------------------------------------------------------------
        run_inference(model, test_data_map[test_number], problem_names[test_number])

        # --------------------------------------------------
        # Loss curve visualization
        x_axis = list(range(0, train_times, 2))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis, loss_plot, color='tab:red',
                label='BGD + Armijo Line Search (BCE Loss)')
        ax.set_xlabel('Epoch ')
        ax.set_ylabel('Loss (BCE) ')
        ax.set_title(
            f"Training Loss Curve — Problem {test_number}: {problem_names[test_number]}\n"
            f"Total Epochs: {train_times}"
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()   # 防止标题/标签被裁剪 / Prevent title/labels from being clipped
        plt.show()