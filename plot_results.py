import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    path = "results/MNIST_FPN_history.pth"
    res_dict = torch.load(path, weights_only=False)
    train_accr = res_dict["train_acc_hist"]
    test_accr = res_dict["test_acc_hist"]
    time_epoch = res_dict["time_hist"]
    avg_mem = res_dict["avg_mem"]
    peak_mem = res_dict["peak_mem"]
    avg_depth = res_dict["avg_depth_hist"]
    max_depth = res_dict["depth_max_hist"]
    epoch = np.arange(len(train_accr)) + 1
    
    #Training and Test accuracy
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Test Accuracy")
    plt.plot(epoch, train_accr, color='k', label="Train Acc.")
    plt.plot(epoch, test_accr, color='g', label="Test Acc.")
    plt.legend()
    plt.show()

    #Runtime
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Runtime per Epoch")
    plt.plot(epoch, time_epoch, color='k')
    plt.show()

    print("Average Memory during run (estimate) in bytes: " + str(avg_mem))
    print("Peak Memory during run in bytes: " + str(peak_mem))

    #Average network depth (number of fixed-point iterations) per epoch
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("# Fixed Point Iterations")
    plt.title("Number of Fixed Point Iterations per Epoch")
    plt.plot(epoch, avg_depth, color='b', label='Average')
    plt.plot(epoch, max_depth, color='k', label='Max')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()