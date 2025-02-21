# Eric Gelphman
# Colorado School of Mines Department of Applied Mathematics and Statistics
# February 20, 2025

from collections import deque
import torch
import matplotlib.pyplot as plt
import time

# Random 5-Layer CV Network using CNN for testing
class RandomLayer(torch.nn.Module):

    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = torch.nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = torch.nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups, n_channels)
        self.norm3 = torch.nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x, d):
        """
        Forward Pass of network
        """
        y = self.norm1(torch.nn.functional.relu(self.conv1(x)))
        return self.norm3(torch.nn.functional.relu(x + self.norm2(d + self.conv2(y))))


def banach(x0, T_eval, data, eps=1.0e-3, max_itr=75):
    """
    Banach Fixed-Point Iteration

    Parameters:
        x0: Initial Guess
        T_eval: Function to evaluate in fixed point iteration
        data: Data in NN
        eps: Epsilon for convergence check
        max_itr: Max. number of iterations

    Return:
        Fixed point of T_eval
    """
    x_k = x0
    x_k1 = T_eval(x0, data)
    k = 0
    res = []
    res_k = (x_k1 - x_k).norm().item() / (1.0e-9 + x_k1.norm().item())
    while (res_k > eps and k < max_itr):
        res.append(res_k)
        x_k = x_k1
        x_k1 = T_eval(x_k, data)
        k += 1
        res_k = (x_k1 - x_k).norm().item() / (1.0e-9 + x_k1.norm().item())

    return x_k, k, res

def f(x):
    return 8.0 + 0.4*torch.sin(2.0*x)

def anderson(x0, T_eval, data, m=5, beta = 0.5, lam=1.0e-6, eps=1.0e-3, max_itr=75):
    """
    Fixed-Point Iteration with Anderson acceleration 

    Parameters:
        x0: Initial guess
        T_eval: Function to evaluate in fixed point iteration
        data: Data in NN
        m: Number of previous iterations to use in least-squares 
           optimization problem
        beta: Parameter in Anderson acceleration iteration
        lam: Regularization parameter
        eps: Epsilon for convergence check

    Return:
        Fixed point of T_eval
    """
    batch_sz, d, h, w = x0.shape
    x_hist = torch.zeros(batch_sz, m, d*h*w, dtype=x0.dtype, device=x0.device)
    T_eval_hist = torch.zeros(batch_sz, m, d*h*w, dtype=x0.dtype, device=x0.device)
    x_hist[:,0] = x0.view(batch_sz, -1)
    T_eval_hist[:,0] = T_eval(x0, data).view(batch_sz,-1)
    x_hist[:,1] = T_eval_hist[:,0]
    T_eval_hist[:,1] = T_eval(T_eval_hist[:,0].view_as(x0), data).view(batch_sz,-1)
    H = torch.zeros(batch_sz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = 1
    H[:,1:,0] = 1
    Batch_RHS = torch.zeros(batch_sz, m+1, 1, dtype=x0.dtype, device=x0.device)
    Batch_RHS[:,0] = 1 

    res = []
    res_k = ((T_eval_hist[:,0] - x_hist[:,0]).norm().item()) / (1.0e-9 + T_eval_hist[:,0].norm().item())
    res.append(res_k)
    k = 1
    res_k = ((T_eval_hist[:,k%m] - x_hist[:,k%m]).norm().item()) / (1.0e-9 + T_eval_hist[:,k%m].norm().item())
    res.append(res_k)
    k += 1
    while (res_k > eps and k < max_itr):
        M = min(k,m)
        G = T_eval_hist[:,:M] - x_hist[:,:M]
        H[:,1:(M+1),1:(M+1)] = torch.bmm(G, G.transpose(1,2)) + lam*torch.eye(M, dtype=x0.dtype, device=x0.device)[None]

        #Solve for alpha
        alpha = None
        try:
            alpha = torch.linalg.solve(H[:,:(M+1),:(M+1)], Batch_RHS[:,:(M+1)])[:,1:(M+1),0]#Result is batch_sz x n
        except RuntimeError:#If matrix is singular solve using QR least squares
            alpha = torch.linalg.lstsq(H[:,:(M+1),:(M+1)], Batch_RHS[:,:(M+1)])[0][:,1:(M+1)]

        #Update data structures
        x_hist[:,k%m] = (1-beta)*((alpha[:,None]@x_hist[:,:M])[:,0]) + beta*((alpha[:,None]@T_eval_hist[:,:M])[:,0])
        T_eval_hist[:,k%m] = T_eval(x_hist[:,k%m].view_as(x0), data).view(batch_sz, -1)
        res_k = ((T_eval_hist[:,k%m] - x_hist[:,k%m]).norm().item()) / (1.0e-9 + T_eval_hist[:,k%m].norm().item())
        res.append(res_k)
        k += 1

    return x_hist[:,k%m].view_as(x0), k, res

def main():
    batch_size = 100
    X = 100.0*torch.randn(batch_size,64,32,32)
    R_NN = RandomLayer(64,128)

    b_start = time.time()
    b_fixed_pt, b_itr, res_b = banach(torch.zeros_like(X), R_NN.forward, X, eps=1.0e-3)
    b_end = time.time()
    print("Execution time for Banach: " + str(b_end - b_start) + "   Number of Banach iterations: " + str(b_itr))
    a_start = time.time()
    a_fixed_pt, a_itr, res_a = anderson(torch.zeros_like(X), R_NN.forward, X, beta=1.5, eps=1.0e-3)
    a_end = time.time()
    print("Execution time for Anderson: " + str(a_end - a_start) + "   Number of Anderson iterations: " + str(a_itr))
    plt.figure()
    plt.semilogy(range(b_itr), res_b, label="Banach", color='b')
    plt.semilogy(range(a_itr), res_a, label="Anderson", color='g')
    plt.xlabel("Iteration")
    plt.ylabel("Relative residual")
    plt.title("Batch Size: " + str(batch_size))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
