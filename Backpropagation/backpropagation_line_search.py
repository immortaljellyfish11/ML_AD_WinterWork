import math
import random
import copy
import pickle

class NeuralNetworkLineSearch:
    def __init__(self, layersize, use_residual=False):
        self.layersize = layersize   # multilayer structure
        self.layer_num = len(layersize)
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []  # Store pre-activation values for ReLU
        self.use_residual = use_residual

        # Initialize weights and biases
        for i in range(self.layer_num-1):
            weight_matrix = [[random.uniform(-0.5,0.5) for _ in range(self.layersize[i])] 
                               for _ in range(self.layersize[i+1])]
            bias_vector = [random.uniform(-0.1,0.1) for _ in range(self.layersize[i+1])]
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, x):
        if x < -500:
            return 0.0
        if x > 500:
            return 1.0
        return 1 / (1 + math.exp(-x))
    
    def dsigmoid(self, a):
        return a * (1 - a)
    
    def relu(self, x):
        beta = 1e-8
        return max(0.0, x)
    
    def drelu(self, z):
        beta = 1e-8
        return 1.0 if z > 0 else 0.0
    
    def forward(self, input_data):
        current_input = input_data
        self.activations = [input_data]
        self.z_values = []
        
        for i in range(self.layer_num-1):
            next_activation = []
            layer_z = []
            
            for j in range(self.layersize[i+1]):
                z = self.biases[i][j]
                for k in range(len(current_input)):
                    z += current_input[k] * self.weights[i][j][k]
                
                layer_z.append(z)
                if i < self.layer_num - 2:
                    relu_out = self.relu(z)
                    # 残差连接：仅当前后两层维度相同时才加 skip connection
                    if self.layersize[i] == self.layersize[i+1] and self.use_residual:
                        activation = relu_out + self.activations[-1][j]
                    else:
                        activation = relu_out
                else:
                    activation = self.sigmoid(z)
                
                next_activation.append(activation)
            
            self.z_values.append(layer_z)
            self.activations.append(next_activation)
            current_input = next_activation
        
        return self.activations[-1]

    def compute_loss(self, input_data, target):
        """BSE Loss: L = -1/N * sum(y*log(a) + (1-y)*log(1-a))"""
        output = self.forward(input_data)
        loss = 0
        for i in range(len(output)):
            a = min(max(output[i], 1e-15), 1 - 1e-15)
            y = target[i]
            loss += -(y * math.log(a) + (1 - y) * math.log(1 - a))
        loss = loss / len(output)
        return loss
    
    def compute_gradients(self, target):
        """
        计算损失函数对每个权重和偏置的梯度。
        梯度本身就代表了“每一个权重对输出值（进而对Loss）的影响程度”。
        """
        output = self.activations[-1]
        
        # 初始化梯度结构
        grad_weights = []
        grad_biases = []
        for i in range(self.layer_num - 1):
            grad_weights.append([[0.0 for _ in range(self.layersize[i])] for _ in range(self.layersize[i+1])])
            grad_biases.append([0.0 for _ in range(self.layersize[i+1])])
            
        # 1. 计算输出层的误差项 (delta)
        # for output layer, Sigmoid + BCE Loss，输出层的 delta 简化为: output - target
        # dLoss/dZ = output - target
        deltas = []
        for j in range(len(output)):
            delta = output[j] - target[j]
            deltas.append(delta)
            
            # 计算输出层权重和偏置的梯度
            for k in range(self.layersize[-2]):
                grad_weights[-1][j][k] = delta * self.activations[-2][k]
            grad_biases[-1][j] = delta
            
        # 2. 反向传播误差项到隐藏层
        current_deltas = deltas
        # prev_error_terms: 上一轮迭代保存的 dL/da（仅当该层有跳跃连接时才非空）
        # 用于将残差路径梯度注入当前层: dL/da[l] += dL/da[l+1] （恒等捷径的偏导=1）
        prev_error_terms = []

        for l in range(self.layer_num - 3, -1, -1):
            layer_deltas = []
            cur_error_terms = []
            has_skip = self.use_residual and self.layersize[l] == self.layersize[l+1]

            for i in range(self.layersize[l+1]):
                # 主路径：dL/da[l+1][i] = sum_j( dL/dz[l+2][j] * w[l+1][j][i] )
                error_term = 0
                for j in range(self.layersize[l+2]):
                    error_term += current_deltas[j] * self.weights[l+1][j][i]

                # 残差路径：若上一层与当前层之间有跳跃连接，则叠加上一层传来的 dL/da
                # 即：dL/da[l+1][i] += dL/da[l+2][i]（来自上次迭代保存的 error_term）
                if prev_error_terms and i < len(prev_error_terms):
                    error_term += prev_error_terms[i]

                delta = error_term * self.drelu(self.z_values[l][i])
                layer_deltas.append(delta)
                cur_error_terms.append(error_term)  # 保存 dL/da[l+1] 供下轮迭代使用

                # 计算隐藏层权重和偏置的梯度
                for k in range(self.layersize[l]):
                    grad_weights[l][i][k] = delta * self.activations[l][k]
                grad_biases[l][i] = delta

            # 若本层存在跳跃连接，则向下传递 dL/da 用于下一层的残差梯度叠加
            prev_error_terms = cur_error_terms if has_skip else []
            current_deltas = layer_deltas
            
        return grad_weights, grad_biases

    def backtracking_line_search(self, input_data, target, grad_weights, grad_biases, initial_alpha=1.0, rho=0.6, c=1e-4):
        """
        Armijo rules (Backtracking Line Search).
        f(x + alpha * d) = f(x) + c * alpha * <grad, d> 
        """
        # Loss: f(x)
        current_loss = self.compute_loss(input_data, target)
        
        # <grad, d> = ||grad||^2
        grad_norm = 0.0
        for l in range(self.layer_num - 1):
            for i in range(self.layersize[l+1]):
                grad_norm += grad_biases[l][i]**2
                for k in range(self.layersize[l]):
                    grad_norm += grad_weights[l][i][k]**2
                    
        alpha = initial_alpha
        
        orig_weights = copy.deepcopy(self.weights)
        orig_biases = copy.deepcopy(self.biases)
        
        max_iters = 25  # 防止死循环的最大迭代次数
        for _ in range(max_iters):
            # 尝试更新权重: W_new = W - alpha * grad
            for l in range(self.layer_num - 1):
                for i in range(self.layersize[l+1]):
                    self.biases[l][i] = orig_biases[l][i] - alpha * grad_biases[l][i]
                    for k in range(self.layersize[l]):
                        self.weights[l][i][k] = orig_weights[l][i][k] - alpha * grad_weights[l][i][k]
                        
            # 计算更新后的 Loss: f(x - alpha * grad)
            new_loss = self.compute_loss(input_data, target)
            
            # 检查 Armijo 准则:
            # f(x - alpha * grad) <= f(x) - c * alpha * ||grad||^2
            if new_loss <= current_loss - c * alpha * grad_norm:
                break # 满足充分下降条件，跳出循环
                
            # 否则，减小步长
            alpha *= rho
            
        # 恢复原始权重（实际的更新将在 train 方法中进行）
        self.weights = orig_weights
        self.biases = orig_biases
        
        return alpha

    def train_with_line_search(self, input_data, target, initial_alpha=1.0):
        """
        使用回溯线搜索进行单步训练。
        """
        # 1. 前向传播，获取当前的激活值
        self.forward(input_data)
        
        # 2. 计算梯度（检查每一个权重对输出/Loss的影响）
        grad_weights, grad_biases = self.compute_gradients(target)
        
        # 3. 使用回溯线搜索寻找最优步长 alpha
        alpha = self.backtracking_line_search(input_data, target, grad_weights, grad_biases, initial_alpha=initial_alpha)
        
        # 4. 使用找到的步长 alpha 更新权重和偏置
        for l in range(self.layer_num - 1):
            for i in range(self.layersize[l+1]):
                self.biases[l][i] -= alpha * grad_biases[l][i]
                for k in range(self.layersize[l]):
                    self.weights[l][i][k] -= alpha * grad_weights[l][i][k]
                    
        # 返回更新后的 Loss 和使用的步长
        final_loss = self.compute_loss(input_data, target)
        return final_loss, alpha

    def compute_batch_loss(self, inputs, targets):
        """计算整个批次的平均 Loss"""
        total_loss = 0
        for i in range(len(inputs)):
            total_loss += self.compute_loss(inputs[i], targets[i])
        return total_loss / len(inputs)

    def compute_batch_gradients(self, inputs, targets):
        """计算整个批次的平均梯度"""
        # 初始化批次梯度结构
        batch_grad_weights = []
        batch_grad_biases = []
        for i in range(self.layer_num - 1):
            batch_grad_weights.append([[0.0 for _ in range(self.layersize[i])] for _ in range(self.layersize[i+1])])
            batch_grad_biases.append([0.0 for _ in range(self.layersize[i+1])])
            
        # 累加每个样本的梯度
        for idx in range(len(inputs)):
            self.forward(inputs[idx])
            grad_weights, grad_biases = self.compute_gradients(targets[idx])
            
            for l in range(self.layer_num - 1):
                for i in range(self.layersize[l+1]):
                    batch_grad_biases[l][i] += grad_biases[l][i]
                    for k in range(self.layersize[l]):
                        batch_grad_weights[l][i][k] += grad_weights[l][i][k]
                        
        # 求平均
        num_samples = len(inputs)
        for l in range(self.layer_num - 1):
            for i in range(self.layersize[l+1]):
                batch_grad_biases[l][i] /= num_samples
                for k in range(self.layersize[l]):
                    batch_grad_weights[l][i][k] /= num_samples
                    
        return batch_grad_weights, batch_grad_biases

    def backtracking_line_search_batch(self, inputs, targets, grad_weights, grad_biases, initial_alpha=1.0, rho=0.5, c=1e-4):
        """基于全批次 Loss 的回溯线搜索"""
        current_loss = self.compute_batch_loss(inputs, targets)
        
        grad_sq_norm = 0.0
        for l in range(self.layer_num - 1):
            for i in range(self.layersize[l+1]):
                grad_sq_norm += grad_biases[l][i]**2
                for k in range(self.layersize[l]):
                    grad_sq_norm += grad_weights[l][i][k]**2
                    
        alpha = initial_alpha
        orig_weights = copy.deepcopy(self.weights)
        orig_biases = copy.deepcopy(self.biases)
        
        max_iters = 25
        for _ in range(max_iters):
            for l in range(self.layer_num - 1):
                for i in range(self.layersize[l+1]):
                    self.biases[l][i] = orig_biases[l][i] - alpha * grad_biases[l][i]
                    for k in range(self.layersize[l]):
                        self.weights[l][i][k] = orig_weights[l][i][k] - alpha * grad_weights[l][i][k]
                        
            new_loss = self.compute_batch_loss(inputs, targets)
            
            if new_loss <= current_loss - c * alpha * grad_sq_norm:
                break
                
            alpha *= rho
            
        self.weights = orig_weights
        self.biases = orig_biases
        
        return alpha

    def train_batch_with_line_search(self, inputs, targets, initial_rate=1.0):
        """
        使用全批量梯度下降 (BGD) 和回溯线搜索进行训练。
        """
        # 1. 计算全批次的平均梯度
        grad_weights, grad_biases = self.compute_batch_gradients(inputs, targets)
        
        # 2. 使用回溯线搜索寻找最优步长 alpha (基于全批次 Loss)
        alpha = self.backtracking_line_search_batch(inputs, targets, grad_weights, grad_biases, initial_alpha=initial_rate)

        # 3. 更新权重和偏置
        for l in range(self.layer_num - 1):
            for i in range(self.layersize[l+1]):
                self.biases[l][i] -= alpha * grad_biases[l][i]
                for k in range(self.layersize[l]):
                    self.weights[l][i][k] -= alpha * grad_weights[l][i][k]
                    
        # 返回更新后的全批次 Loss 和使用的步长
        final_loss = self.compute_batch_loss(inputs, targets)
        return final_loss, alpha

    def reasoning(self, input_data):
        return self.forward(input_data)

    def save(self, filepath):
        """将模型权重保存到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'layersize': self.layersize,
                'weights': self.weights,
                'biases': self.biases,
                'proj_weights': self.weights
            }, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """从文件加载模型权重，返回一个新的 NeuralNetworkLineSearch 实例"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(layersize=data['layersize'])
        model.weights = data['weights']
        model.biases = data['biases']
        model.weights = data['proj_weights']
        print(f"Model loaded from {filepath}")
        return model
    
# 简单的测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nn = NeuralNetworkLineSearch([2, 4, 1])
    X = [0.5, 0.8]
    Y = [0.2]
    training_data_XOR = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]

    epochs = 100
    losses = []
    alphas = []
    
    # 提取全批次的输入和目标
    inputs_XOR = [item[0] for item in training_data_XOR]
    targets_XOR = [item[1] for item in training_data_XOR]

    print("Initial Batch Loss:", nn.compute_batch_loss(inputs_XOR, targets_XOR))
    
    for epoch in range(epochs):
        # 使用全批量更新，初始步长保持 0.5 不变
        loss, alpha = nn.train_batch_with_line_search(inputs_XOR, targets_XOR, initial_rate=20)
        losses.append(loss)
        alphas.append(alpha)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss:.6f}, Step Size (alpha) = {alpha}")

    # 绘制 Loss 和 步长(alpha) 的变化曲线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(range(1, epochs + 1), losses, 'b-', linewidth=2, label='Loss')
    ax1.set_title('Training Loss over Epochs (Backtracking Line Search)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(range(1, epochs + 1), alphas, 'r-', linewidth=2, label='Step Size (Alpha)')
    ax2.set_title('Step Size (\u03b1) over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Alpha')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('line_search_visualization.png')
    print("Visualization saved to 'line_search_visualization.png'")
    plt.show()
