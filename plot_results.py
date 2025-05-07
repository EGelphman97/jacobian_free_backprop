import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    """
    #Row index is Batch Size 0->50, 1->100, 2->150, 3->200, 4->250, 5->400, 6->500, 7->600, 8->1000
    #Column 0 is average time per epoch over first 10 epochs
    #Column 1 is average memory in MB during run
    #Column 2 is peak memory in MB during run
    #Column 3 is average depth in MB during run
    sz_arr = np.array([50,100,150,200,250,400,500,600,1000,1250])

    #INN without AA
    results_INN_noAA = np.zeros((10,4), dtype=np.float32)
    idx = 0
    for sz in sz_arr:
        res_dict = torch.load("results/MNIST_FPN_history_INN_noAA_Batch"+str(sz)+".pth", weights_only=False)
        results_INN_noAA[idx][0] = np.average(np.array(res_dict["time_hist"]))
        results_INN_noAA[idx][1] = res_dict["avg_mem"]/1.0e6
        results_INN_noAA[idx][2] = res_dict["peak_mem"]/1.0e6
        results_INN_noAA[idx][3] = np.average(np.array(res_dict["avg_depth_hist"]))
        idx += 1

    #INN with AA
    results_INN_AA = np.zeros((10,4), dtype=np.float32)
    idx = 0
    for sz in sz_arr:
        print("Batch Size: " + str(sz))
        res_dict = torch.load("results/MNIST_FPN_history_INN_AA_Batch"+str(sz)+".pth", weights_only=False)
        results_INN_AA[idx][0] = np.average(np.array(res_dict["time_hist"]))
        results_INN_AA[idx][1] = res_dict["avg_mem"]/1.0e6
        results_INN_AA[idx][2] = res_dict["peak_mem"]/1.0e6
        results_INN_AA[idx][3] = np.average(np.array(res_dict["avg_depth_hist"]))
        print("Average percentage of Anderson iterations that require least squares solve: " + str(res_dict["avg_pct_ls"]))
        idx += 1

    #Jacobian-Based INN without AA
    results_Jac_noAA = np.zeros((10,4), dtype=np.float32)
    idx = 0
    for sz in sz_arr:
        res_dict = torch.load("results/MNIST_FPN_Jacobian_based_history_noAA_Batch"+str(sz)+".pth", weights_only=False)
        results_Jac_noAA[idx][0] = np.average(np.array(res_dict["time_hist"]))
        results_Jac_noAA[idx][1] = res_dict["avg_mem"]/1.0e6
        results_Jac_noAA[idx][2] = res_dict["peak_mem"]/1.0e6
        results_Jac_noAA[idx][3] = np.average(np.array(res_dict["depth_test_hist"]))
        idx += 1

    #Jacobian-Based INN with AA
    results_Jac_AA = np.zeros((10,4), dtype=np.float32)
    idx = 0
    for sz in sz_arr:
        print("Batch Size: " + str(sz))
        res_dict = torch.load("results/MNIST_FPN_Jacobian_based_history_AA_Batch"+str(sz)+".pth", weights_only=False)
        results_Jac_AA[idx][0] = np.average(np.array(res_dict["time_hist"]))
        results_Jac_AA[idx][1] = res_dict["avg_mem"]/1.0e6
        results_Jac_AA[idx][2] = res_dict["peak_mem"]/1.0e6
        results_Jac_AA[idx][3] = np.average(np.array(res_dict["depth_test_hist"]))
        print("Average percentage of Anderson iterations that require least squares solve: " + str(res_dict["avg_pct_ls"]))
        idx += 1

    #Explicit Network (CNN)
    results_Exp = np.zeros((10,4), dtype=np.float32)
    idx = 0
    for sz in sz_arr:
        res_dict = torch.load("results/MNIST_FPN_Explicit_history_Batch"+str(sz)+".pth", weights_only=False)
        results_Exp[idx][0] = np.average(np.array(res_dict["time_hist"]))
        results_Exp[idx][1] = res_dict["avg_mem"]/1.0e6
        results_Exp[idx][2] = res_dict["peak_mem"]/1.0e6
        idx += 1

    #Mean Runtime per Epoch 
    plt.figure()
    plt.plot(sz_arr, results_INN_noAA[:,0], color="green",label="INN no AA")
    plt.plot(sz_arr, results_INN_AA[:,0], color="blue",label="INN AA")
    plt.plot(sz_arr, results_Jac_noAA[:,0], color="red",label="Jac no AA")
    plt.plot(sz_arr, results_Jac_AA[:,0], color="orange",label="Jac AA")
    plt.plot(sz_arr, results_Exp[:,0], color="gray",label="CNN")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (s)")
    plt.title("Mean Runtime per Epoch")
    plt.legend()
    plt.show()

    #Average Memory in MB
    plt.figure()
    plt.plot(sz_arr, results_INN_noAA[:,1], color="green",label="INN no AA")
    plt.plot(sz_arr, results_INN_AA[:,1], color="blue",label="INN AA")
    #plt.plot(sz_arr, results_Jac_noAA[:,1], color="red",label="Jac no AA")
    #plt.plot(sz_arr, results_Jac_AA[:,1], color="orange",label="Jac AA")
    plt.plot(sz_arr, results_Exp[:,1], color="gray",label="CNN")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Memory (MB)")
    plt.title("Average Memory Usage During Training")
    plt.legend()
    plt.show()

    #Peak Memory in MB
    plt.figure()
    plt.plot(sz_arr, results_INN_noAA[:,2], color="green",label="INN no AA")
    plt.plot(sz_arr, results_INN_AA[:,2], color="blue",label="INN AA")
    #plt.plot(sz_arr, results_Jac_noAA[:,2], color="red",label="Jac no AA")
    #plt.plot(sz_arr, results_Jac_AA[:,2], color="orange",label="Jac AA")
    plt.plot(sz_arr, results_Exp[:,2], color="gray",label="CNN")
    plt.xlabel("Batch Size")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Peak Memory Usage During Training")
    plt.legend()
    plt.show()

    #Average Depth (Number of Fixed Point Iterations) per Epoch
    plt.figure()
    plt.plot(sz_arr, results_INN_noAA[:,3], color="green",label="INN no AA")
    plt.plot(sz_arr, results_INN_AA[:,3], color="blue",label="INN AA")
    plt.plot(sz_arr, results_Jac_noAA[:,3], color="red",label="Jac no AA")
    plt.plot(sz_arr, results_Jac_AA[:,3], color="orange",label="Jac AA")
    plt.xlabel("Batch Size")
    plt.ylabel("Average # Fixed Point Iterations")
    plt.title("Average Number of Fixed Point Iterations per Epoch")
    plt.legend()
    plt.show()
    """
    res_dict = torch.load("results/MNIST_FPN_history_util_noAA.pth", weights_only=False)
    util_noAA = res_dict['util_hist']
    print("Average Memory per Epoch INN without AA: " + str(res_dict['avg_mem']))
    res_dict = torch.load("results/MNIST_FPN_history_util_AA.pth", weights_only=False)
    util_AA = res_dict['util_hist']
    print("Average Memory per Epoch INN with AA: " + str(res_dict['avg_mem']))
    res_dict = torch.load("results/MNIST_FPN_Explicit_history_util.pth", weights_only=False)
    util_Exp = res_dict['util_hist']
    print("Average Memory per Epoch CNN: " + str(res_dict['avg_mem']))
    epoch = range(1,11)
    plt.figure()
    plt.plot(epoch, util_noAA, label="INN no AA", color = 'b')
    plt.plot(epoch, util_AA, label="INN AA", color = 'g')
    plt.plot(epoch, util_Exp, label="CNN", color = 'k')
    plt.xlabel("Epoch")
    plt.ylabel("% Utilization")
    plt.title("RTX 4060 GPU Utilization During Training")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()