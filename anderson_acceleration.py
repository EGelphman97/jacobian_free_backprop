# Eric Gelphman
# Colorado School of Mines Department of Applied Mathematics and Statistics
# February 13, 2025
import torch

def banach(N, T_eval, eps=1.0e-3):
    """
    Banach Fixed-Point Iteration

    Parameters:
        N: Dimension of input vector
        T_eval: Function to evaluate in fixed point iteration
        eps: Epsilon for convergence check

    Return:
        Fixed point of T_eval
    """
    x_k = torch.zeros(N)
    x_k1 = T_eval(x_k)
    k = 0
    while torch.linalg.vector_norm(x_k - x_k1) > eps:
        x_k = x_k1
        x_k1 = T_eval(x_k)
        k += 1
    return x_k, k

def f(x):
    return 8.0 + 0.4*torch.sin(2.0*x)

def anderson(N, T_eval, beta = 0.5, m=5, lam=1.0e-6, eps=1.0e-3):
    """
    Fixed-Point Iteration with Anderson acceleration 

    Parameters:
        N: Dimension of input vector
        T_eval: Function to evaluate in fixed point iteration
        beta: Parameter in Anderson acceleration iteration
        m: Number of previous iterations to use in least-squares 
           optimization problem
        lam: Regularization parameter
        eps: Epsilon for convergence check

    Return:
        Fixed point of T_eval
    """
    x_hist = []#Stack of previous x_k values, highest index is most recent
    G = []#Stack of previous residual T_eval(x_k) - x_k values, highest index is most recent
    T_eval_hist = []#Stack of previous T_eval(x_k) values, highest index is most recent
    x_hist.append(torch.zeros(N))
    G.append(T_eval(x_hist[0]) - x_hist[0])
    k = 0
    T_eval_hist.append(T_eval(x_hist[0]))
    while torch.linalg.vector_norm(x_hist[-1] - T_eval_hist[-1]) > eps:
        if k < m-1:
            #Update x_k using Banach fixed point iteration
            x_k = T_eval(x_hist[-1])
            T_k = T_eval(x_k)
            #Update history stacks
            x_hist.append(x_k)
            T_eval_hist.append(T_k)
            G.append(T_k - x_k)
        else:
            #Solve least-squares problem
            H = torch.ones((m+1,m+1))
            H[0,0] = 0
            GG = torch.stack(G,dim=1)
            y = torch.zeros(m+1)
            y[0] = 1.0
            H[1:,1:] = (GG.T)@GG + lam*torch.eye(m)
            alpha = torch.linalg.solve(H,y)[1:]
            #Update x_k
            s1 = torch.zeros(N)
            s2 = torch.zeros(N)
            for i in range(1,m+1):
                s1 += (alpha[i-1].item())*x_hist[-i]
                s2 += (alpha[i-1].item())*T_eval_hist[-i]
            x_k = (1.0 - beta)*s1 + beta*s2
            T_k = T_eval(x_k)
            #Update history stacks
            x_hist.append(x_k)
            T_eval_hist.append(T_k)
            G.append(T_k - x_k)
            x_ = x_hist.pop(0)
            T_ = T_eval_hist.pop(0)
            G_ = G.pop(0)
        k+=1

    return x_hist[-1], k

def main():
    b_f = banach(25, f, eps=1.0e-4)
    print(b_f)
    a_f = anderson(25, f, eps=1.0e-4)
    print(a_f)

if __name__ == "__main__":
    main()
