import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    path = "results/MNIST_FPN_history.pth"
    res_dict = torch.load(path, weights_only=False)
    train_accr = res_dict["train_acc_hist"]
    test_accr = res_dict["test_acc_hist"]
    time_epoch = res_dict["time_hist"]
    mem_epoch = res_dict["mem_hist"]
    epoch = np.arange(len(train_accr)) + 1
    
    #Training and Test accuracy
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Test Accuracy")
    plt.plot(epoch, train_accr, color='b', label="Train Acc.")
    plt.plot(epoch, test_accr, color='g', label="Test Acc.")
    plt.legend()
    plt.show()

    #Runtime
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Runtime per Epoch")
    plt.plot(epoch, time_epoch, color='b')
    plt.show()

    #Memory Usage
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Memory (MB)")
    plt.title("Memory Usage per Epoch")
    plt.plot(epoch, mem_epoch, color='b')
    plt.show()
    
if __name__ == "__main__":
    main()