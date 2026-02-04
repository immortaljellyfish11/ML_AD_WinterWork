from backpropagation_EG1 import NeuralNetwork
import matplotlib.pyplot as plt

if __name__ == "__main__":
    example = NeuralNetwork(layersize=[2, 4, 1], learning_rate=0.2)
    train_times = int(input("Please input the training times: "))

    # Prob.1:The XOR problem --training data
    training_data_XOR = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    #Prob.2: 二分类任务 -- training data
    training_data_Binary = [
         ([11, 17], [0]),         
         ([13, 17], [1]),
         ([13, 19], [1]),
         ([10, 18], [0]),
         ([14, 18], [1]),
         ([11, 16], [0]),
         ([12, 19], [1]),
         ([14, 15], [0]),
         ([15, 18], [1]),
         ([9, 17], [0]),
         ([16, 17], [1]),
         ([12, 20], [1]),
         ([10, 20], [0]),
         ([15, 16], [1]),
         ([13, 16], [1]),
         ([14, 15], [1])
        ]
    
    # Train 
    loss_plot = []
    loss = 0
    for i in range(train_times):
        for inputs, target in training_data_XOR:  
            loss += example.train(inputs, target)
        if i % 100 == 0:
            loss_plot.append(loss / len(training_data_XOR))
        if i % 1000 ==0:
            print(f"We've trained {i} times.")
        loss = 0
         
    # show a figure of loss
    plt.plot(range(0, train_times, 100), loss_plot)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.show()

    # Test
    print("\nTest results:")
    for input, _ in training_data_XOR:
        output = example.reasoning(input)
        print(f"Input: {input} --> Output: {output[0]:.4f}")