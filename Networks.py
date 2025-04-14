import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

classification = torch.tensor
latent_variable = torch.tensor
image = torch.tensor

def anderson(net, u0, Qd, tol=1.0e-3, max_iters=100, m=5, beta=0.5, lam=1.0e-6):
        """
        Fixed-Point Iteration with Anderson acceleration 

        Parameters:
            net: INN object that has implementation of function for 
                 evaluating operator T latent_space_forward()
            u0: Initial guess
            Qd: Data mapped to latent space
            tol: Error tolerance for convergence
            m: Number of previous iterations to use in least-squares 
               optimization problem
            beta: Parameter in Anderson acceleration iteration
            lam: Regularization parameter

        Return:
            Fixed point of T_eval, value of u after previous iteration, and number of iterations
        """
        batch_sz, d, h, w = u0.shape
        u_hist = torch.zeros(batch_sz, m, d*h*w, dtype=u0.dtype, device=u0.device)
        T_hist = torch.zeros(batch_sz, m, d*h*w, dtype=u0.dtype, device=u0.device)
        u_hist[:,0] = u0.view(batch_sz, -1)
        T_hist[:,0] = net.latent_space_forward(u0, Qd).view(batch_sz,-1)
        u_hist[:,1] = T_hist[:,0]
        T_hist[:,1] = net.latent_space_forward(T_hist[:,0].view_as(u0), Qd).view(batch_sz,-1)
        H = torch.zeros(batch_sz, m+1, m+1, dtype=u0.dtype, device=u0.device)
        H[:,0,1:] = 1
        H[:,1:,0] = 1
        Batch_RHS = torch.zeros(batch_sz, m+1, 1, dtype=u0.dtype, device=u0.device)
        Batch_RHS[:,0] = 1 

        res_k = ((T_hist[:,0] - u_hist[:,0]).norm().item()) / (1.0e-9 + T_hist[:,0].norm().item())
        k = 1
        res_k = ((T_hist[:,k%m] - u_hist[:,k%m]).norm().item()) / (1.0e-9 + T_hist[:,k%m].norm().item())
        k += 1
        while (res_k > tol and k < max_iters):
            M = min(k,m)
            G = T_hist[:,:M] - u_hist[:,:M]
            H[:,1:(M+1),1:(M+1)] = torch.bmm(G, G.transpose(1,2)) + lam*torch.eye(M, dtype=u0.dtype, device=u0.device)[None]

            #Solve for alpha
            alpha = None
            try:
                alpha = torch.linalg.solve(H[:,:(M+1),:(M+1)], Batch_RHS[:,:(M+1)])[:,1:(M+1),0]#Result is batch_sz x n
            except RuntimeError:#If matrix is singular solve using Householder QR least squares
                alpha = torch.linalg.lstsq(H[:,:(M+1),:(M+1)], Batch_RHS[:,:(M+1)])[0][:,1:(M+1)]

            #Update data structures
            u_hist[:,k%m] = (1.0-beta)*((alpha[:,None]@u_hist[:,:M])[:,0]) + beta*((alpha[:,None]@T_hist[:,:M])[:,0])
            T_hist[:,k%m] = net.latent_space_forward(u_hist[:,k%m].view_as(u0), Qd).view(batch_sz, -1)
            res_k = ((T_hist[:,k%m] - u_hist[:,k%m]).norm().item()) / (1.0e-9 + T_hist[:,k%m].norm().item())
            k += 1

        return u_hist[:,k%m].view_as(u0), u_hist[:,(k-1)%m].view_as(u0), k

def forward_implicit(net, d: image, eps=1.0e-3, max_depth=100,
                     depth_warning=False):
    ''' Fixed Point Iteration forward prop

        With gradients detached, find fixed point. During forward iteration,
        u is updated via R(u,Q(d)) and Lipschitz constant estimates are
        refined. Gradient are attached performing one final step.
    '''
    with torch.no_grad():
        net.depth = 0.0
        Qd = net.data_space_forward(d)
        u = torch.zeros(Qd.shape, device=net.device())
        u_prev = np.Inf*torch.ones(u.shape, device=net.device())
        if not net.use_anderson:
            all_samp_conv = False
            while not all_samp_conv and net.depth < max_depth:
                u_prev = u.clone()
                u = net.latent_space_forward(u, Qd)
                res_norm = torch.max(torch.norm(u - u_prev, dim=1))
                net.depth += 1.0
                all_samp_conv = res_norm <= eps
        else:#Anderson acceleration  
            u, u_prev, num_itr = anderson(net, u, Qd, tol=eps, max_iters=max_depth, beta=1.5)
            net.depth += num_itr

        if net.training:
            net.normalize_lip_const(u_prev, Qd)

    if net.depth >= max_depth and depth_warning:
        print("\nWarning: Max Depth Reached - Break Forward Loop\n")

    attach_gradients = net.training
    if attach_gradients:
        Qd = net.data_space_forward(d)
        return net.map_latent_to_inference(
            net.latent_space_forward(u.detach(), Qd))
    else:
        return net.map_latent_to_inference(u).detach()


def forward_explicit(net, d: image, eps=1.0e-3, max_depth=100,
                     depth_warning=False):
    '''
        Apply Explicit Forward Propagation
    '''

    net.depth = 0.0

    Qd = net.data_space_forward(d)
    u = torch.zeros(Qd.shape, device=net.device())
    Ru = net.latent_space_forward(u, Qd)

    return net.map_latent_to_inference(Ru)


def normalize_lip_const(net, u: latent_variable, v: latent_variable):
    ''' Scale convolutions in R to make it gamma Lipschitz

        It should hold that |R(u,v) - R(w,v)| <= gamma * |u-w| for all u
        and w. If this doesn't hold, then we must rescale the convolution.
        Consider R = I + Conv. To rescale, ideally we multiply R by

            norm_fact = gamma * |u-w| / |R(u,v) - R(w,v)|,

        averaged over a batch of samples, i.e. R <-- norm_fact * R. The
        issue is that ResNets include an identity operation, which we don't
        wish to rescale. So, instead we use

            R <-- I + norm_fact * Conv,

        which is accurate up to an identity term scaled by (norm_fact - 1).
        If we do this often enough, then norm_fact ~ 1.0 and the identity
        term is negligible.
    '''
    noise_u = torch.randn(u.size(), device=net.device())
    noise_v = torch.randn(u.size(), device=net.device())
    w = u.clone() + noise_u
    Rwv = net.latent_space_forward(w, v + noise_v)
    Ruv = net.latent_space_forward(u, v + noise_v)
    R_diff_norm = torch.mean(torch.norm(Rwv - Ruv, dim=1))
    u_diff_norm = torch.mean(torch.norm(w - u, dim=1))
    R_is_gamma_lip = R_diff_norm <= net.gamma * u_diff_norm
    if not R_is_gamma_lip:
        violation_ratio = net.gamma * u_diff_norm / R_diff_norm
        normalize_factor = violation_ratio ** (1.0 / net._lat_layers)
        for i in range(net._lat_layers):
            net.latent_convs[i][0].weight.data *= normalize_factor
            net.latent_convs[i][0].bias.data *= normalize_factor
            net.latent_convs[i][3].weight.data *= normalize_factor
            net.latent_convs[i][3].bias.data *= normalize_factor


# ------------------------------------------------------------------------------------------------
# MNIST Architecture
# ------------------------------------------------------------------------------------------------

class MNIST_FPN(nn.Module):
    def __init__(self, lat_layers=4, num_channels=32, contraction_factor=0.1,
                 momentum=0.1, architecture='FPN', use_and=False):
        super().__init__()

        self._channels = num_channels
        self._lat_layers = lat_layers
        self.gamma = contraction_factor
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.drop_outR = nn.Dropout2d(0.0)
        self.drop_outS = nn.Dropout2d(0.2)
        self.mom = momentum
        self.max_pool = nn.MaxPool2d(kernel_size=3)
        self.architecture = architecture
        self.depth = 0.0

        self.channel_dim = 32

        self.conv_d1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.conv_d2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,
                                 bias=True)

        self.latent_convs = nn.ModuleList([nn.Sequential(
                                           nn.Conv2d(in_channels=num_channels,
                                                     out_channels=num_channels,
                                                     kernel_size=3, stride=1,
                                                     padding=(1, 1)),
                                           self.leaky_relu,
                                           self.drop_outR,
                                           nn.Conv2d(in_channels=num_channels,
                                                     out_channels=num_channels,
                                                     kernel_size=3, stride=1,
                                                     padding=(1, 1)),
                                           self.leaky_relu,
                                           self.drop_outR)
                                           for _ in range(lat_layers)])

        self.conv_y = nn.Conv2d(in_channels=self.channel_dim,
                                out_channels=16, kernel_size=3, stride=1)
        self.fc_y = nn.Linear(784, 10, bias=False)

        self.bn_1 = nn.BatchNorm2d(16)
        self.bn_2 = nn.BatchNorm2d(self.channel_dim)
        self.lat_batch_norm = nn.ModuleList([nn.BatchNorm2d(num_channels,
                                                            momentum=self.mom,
                                                            affine=False)
                                            for _ in range(lat_layers)])
        self.use_anderson = use_and

    def name(self):
        '''
            Assign name to model depending on the architecture.

            This is useful for saving the model and choosing which forward
            prop to use.
        '''
        if self.architecture == 'FPN':
            return 'MNIST_FPN'
        elif self.architecture == 'Jacobian':
            return 'MNIST_FPN_Jacobian_based'
        else:
            return 'MNIST_FPN_Explicit'

    def device(self):
        return next(self.parameters()).data.device

    def data_space_forward(self, d: image) -> latent_variable:
        ''' Transform images into feature vectors in latent space

            The data space operator does *not* need to be 1-Lipschitz; however,
            bounding the singular values can improve generalization. A
            multiplicative factor is added in each update to control the
            Lipschitz constant.
        '''

        Qd = self.max_pool(self.leaky_relu(self.bn_1(self.conv_d1(d))))
        Qd = self.leaky_relu(self.bn_2(self.conv_d2(Qd)))
        return Qd

    def latent_space_forward(self, u: latent_variable, v: latent_variable):
        ''' Fixed point operator on latent space (when v is fixed)

            R(u,v) is used in fixed point iteration of FPN to find u*
            satisfying
            u* = R(u*, v). To make R be a contraction in u, we estimate a
            Lipschitz constant and normalize updates using this.
        '''
        uv = u + v
        for idx, conv in enumerate(self.latent_convs):
            res = (self.leaky_relu(conv(uv)))
            if self.architecture == 'Jacobian':
                uv = uv + res
            else:
                uv = self.lat_batch_norm[idx](uv + res)
        R_uv = self.gamma * uv
        return R_uv

    def map_latent_to_inference(self, u: latent_variable) -> classification:
        ''' Transform feature vectors into a classification

            This is the final step of FPN, which flattens and
            then applies affine mappings to input. Operations do *not* need to
            be 1-Lipschitz.
        '''
        u = self.drop_outS(self.leaky_relu(self.conv_y(u)))
        n_samples = u.shape[0]
        u = u.view(n_samples, -1)
        y = self.fc_y(u)
        return y

    def forward(self, d: image, eps=1.0e-3, max_depth=100,
                depth_warning=False) -> classification:
        ''' FPN forward prop

            With gradients detached, find fixed point. During forward
            iteration,
            u is updated via R(u,Q(d)) and Lipschitz constant estimates are
            refined. Gradient are attached performing one final step.
        '''

        if self.architecture == 'Explicit':
            return forward_explicit(self, d, eps=eps, max_depth=max_depth,
                                    depth_warning=False)
        else:
            return forward_implicit(self, d, eps=eps, max_depth=max_depth,
                                    depth_warning=False)

    def normalize_lip_const(self, u: latent_variable, v: latent_variable):
        return normalize_lip_const(self, u, v)


# -----------------------------------------------------------------------------
# SVHN Architectures
# -----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """
        Block architecture borrowed for ResNets. Each block defines the
        ResNet operator.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.drop_out = nn.Dropout2d(0.1)
        self.conv_d1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_d2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.leakyrelu = nn.LeakyReLU(0.1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0, planes//4,
                                                  planes//4),
                                                  "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes,
                               kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.drop_out(self.leakyrelu(self.bn1(self.conv_d1(x))))
        out = self.drop_out(self.bn2(self.conv_d2(out)))
        out += self.shortcut(x)
        out = self.leakyrelu(out)
        return out


def _weights_init(m):
    """
        Initialize weights as in KaimingHe 2015
    """
    # classname = m.__class__.__name__
    # if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #    init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def _make_layer(net, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
        layers.append(block(net.in_planes, planes, stride))
        net.in_planes = planes * block.expansion

    return nn.Sequential(*layers)


class SVHN_FPN(nn.Module):
    def __init__(self, lat_layers=4, num_channels=32, contraction_factor=0.1,
                 momentum=0.1, block=BasicBlock, num_blocks=[1, 1, 1],
                 architecture='FPN', use_and=False):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self._channels = num_channels
        self._lat_layers = lat_layers
        self.gamma = contraction_factor
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.drop_outR = nn.Dropout2d(0.0)
        self.drop_outS = nn.Dropout2d(0.2)
        self.mom = momentum
        self.architecture = architecture
        self.depth = 0.0

        self.channel_dim = 64

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.latent_convs = nn.ModuleList([nn.Sequential(
                                           nn.Conv2d(in_channels=num_channels,
                                                     out_channels=num_channels,
                                                     kernel_size=3, stride=1,
                                                     padding=(1, 1)),
                                           self.leaky_relu,
                                           self.drop_outR,
                                           nn.Conv2d(in_channels=num_channels,
                                                     out_channels=num_channels,
                                                     kernel_size=3, stride=1,
                                                     padding=(1, 1)),
                                           self.leaky_relu,
                                           self.drop_outR)
                                           for _ in range(lat_layers)])

        self.conv_y = nn.Conv2d(in_channels=self.channel_dim,
                                out_channels=16, kernel_size=3, stride=1)
        self.fc_y = nn.Linear(576, 10, bias=False)

        self.bn_1 = nn.BatchNorm2d(self.channel_dim)
        self.bn_2 = nn.BatchNorm2d(self.channel_dim)
        self.lat_batch_norm = nn.ModuleList([nn.BatchNorm2d(num_channels,
                                                            momentum=self.mom,
                                                            affine=False)
                                            for _ in range(lat_layers)])
        self.use_anderson = use_and

    def name(self):
        '''
            Assign name to model depending on the architecture.

            This is useful for saving the model and choosing which forward
            prop to use.
        '''
        if self.architecture == 'FPN':
            return 'SVHN_FPN'
        elif self.architecture == 'Jacobian':
            return 'SVHN_FPN_Jacobian_based'
        else:
            return 'SVHN_FPN_Explicit'

    def device(self):
        return next(self.parameters()).data.device

    def data_space_forward(self, d: image) -> latent_variable:
        ''' Transform images into feature vectors in latent space

            The data space operator does *not* need to be 1-Lipschitz; however,
            bounding the singular values can improve generalization. A
            multiplicative factor is added in each update to control the
            Lipschitz constant.
        '''
        Qd = self.leaky_relu(self.bn1(self.conv1(d)))
        Qd = self.layer1(Qd)
        Qd = self.layer2(Qd)
        Qd = self.layer3(Qd)
        return Qd

    def latent_space_forward(self, u: latent_variable, v: latent_variable):
        ''' Fixed point operator on latent space (when v is fixed)

            R(u,v) is used in fixed point iteration of FPN to find u*
            satisfying
            u* = R(u*, v). To make R be a contraction in u, we estimate a
            Lipschitz constant and normalize updates using this.
        '''
        if self.architecture == 'Jacobian':
            uv = u + v
            for idx, conv in enumerate(self.latent_convs):
                res = (self.leaky_relu(conv(uv)))
                uv = uv + res
            R_uv = self.gamma * uv
            return R_uv
        else:
            uv = u + v
            for idx, conv in enumerate(self.latent_convs):
                res = (self.leaky_relu(conv(uv)))
                uv = self.lat_batch_norm[idx](uv + res)
            R_uv = self.gamma * uv
            return R_uv

    def map_latent_to_inference(self, u: latent_variable) -> classification:
        ''' Transform feature vectors into a classification

            This is the final step of FPN, which flattens and
            then applies affine mappings to input. Operations do *not* need to
            be 1-Lipschitz.
        '''

        u = self.drop_outS(self.leaky_relu(self.conv_y(u)))
        n_samples = u.shape[0]
        u = u.view(n_samples, -1)
        y = self.fc_y(u)
        return y

    def forward(self, d: image, eps=1.0e-3, max_depth=100,
                depth_warning=False) -> classification:
        ''' FPN forward prop

            With gradients detached, find fixed point. During forward
            iteration,
            u is updated via R(u,Q(d)) and Lipschitz constant estimates are
            refined. Gradient are attached performing one final step.
        '''

        if self.architecture == 'Explicit':
            return forward_explicit(self, d, eps=eps, max_depth=max_depth,
                                    depth_warning=False)
        else:
            return forward_implicit(self, d, eps=eps, max_depth=max_depth,
                                    depth_warning=False)

    def normalize_lip_const(self, u: latent_variable, v: latent_variable):
        return normalize_lip_const(self, u, v)

    def _make_layer(self, block, planes, num_blocks, stride):

        return _make_layer(self, block, planes, num_blocks, stride)


# -----------------------------------------------------------------------------
# CIFAR10 Architectures
# -----------------------------------------------------------------------------
class CIFAR10_FPN(nn.Module):
    def __init__(self, data_layers=16, num_channels=35, contraction_factor=0.5,
                 momentum=0.1, lat_layers=5, architecture='FPN', use_and=False):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self._channels = num_channels
        self._data_layers = data_layers
        self.gamma = contraction_factor
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.drop_outQ = nn.Dropout2d(1.5e-1)
        self.drop_outR = nn.Dropout2d(0.0)
        self.drop_outS = nn.Dropout(1.0e-1)
        self._lat_layers = lat_layers
        self.mom = momentum
        self.depth = 0.0

        def in_chan(i):
            return 35

        def out_chan(i):
            if i == (2 * data_layers - 1):
                return num_channels
            else:
                return in_chan(i)

        self.pad = nn.ConstantPad2d(4, 0.449)
        self.label_fc = nn.Linear(25 * num_channels, 10, bias=False)
        self.architecture = architecture

        self.data_conv_d = nn.Conv2d(in_channels=3,
                                     out_channels=in_chan(0),
                                     kernel_size=7, stride=1,
                                     padding=(3, 3), padding_mode='replicate')

        self.data_convs = nn.ModuleList([nn.Sequential(
                            nn.Conv2d(in_channels=in_chan(i),
                                      out_channels=out_chan(i),
                                      kernel_size=3, stride=1,
                                      padding=(1, 1)),
                            self.leaky_relu,
                            self.drop_outQ,
                            nn.Conv2d(in_channels=out_chan(i),
                                      out_channels=out_chan(i),
                                      kernel_size=3, stride=1,
                                      padding=(1, 1)))
                            for i in range(2 * data_layers)])

        self.latent_convs = nn.ModuleList([nn.Sequential(
                                nn.Conv2d(in_channels=num_channels,
                                          out_channels=num_channels,
                                          kernel_size=3, stride=1,
                                          padding=(1, 1)),
                                self.leaky_relu,
                                self.drop_outR,
                                nn.Conv2d(in_channels=num_channels,
                                          out_channels=num_channels,
                                          kernel_size=3, stride=1,
                                          padding=(1, 1)))
                                for _ in range(lat_layers)])

        self.dat_batch_norm = nn.ModuleList([nn.BatchNorm2d(out_chan(i),
                                                            momentum=self.mom,
                                                            affine=False)
                                            for i in range(2 * data_layers)])

        self.lat_batch_norm = nn.ModuleList([nn.BatchNorm2d(num_channels,
                                                            momentum=self.mom,
                                                            affine=False)
                                            for _ in range(lat_layers)])
        self.conv_y = nn.Conv2d(num_channels, num_channels,
                                kernel_size=3, stride=2, padding=(1, 1))
        self.use_anderson = use_and

    def name(self):
        '''
            Assign name to model depending on the architecture.

            This is useful for saving the model and choosing which forward
            prop to use.
        '''
        if self.architecture == 'FPN':
            return 'CIFAR10_FPN'
        elif self.architecture == 'FPN_Unaugmented':
            return 'CIFAR10_FPN_Unaugmented'
        elif self.architecture == 'Jacobian_Unaugmented':
            return 'CIFAR10_FPN_Jacobian_based_Unaugmented'
        elif self.architecture == 'Jacobian':
            return 'CIFAR10_FPN_Jacobian_based'
        elif self.architecture == 'Explicit_Unaugmented':
            return 'CIFAR10_FPN_Explicit_Unaugmented'
        else:
            return 'CIFAR10_FPN_Explicit'

    def device(self):
        return next(self.parameters()).data.device

    def data_space_forward(self, d: image) -> latent_variable:
        ''' Transform images into feature vectors in latent space

            The data space operator does *not* need to be 1-Lipschitz; however,
            bounding the singular values can improve generalization. A
            multiplicative factor is added in each update to control the
            Lipschitz constant.
        '''
        u = self.leaky_relu(self.data_conv_d(self.pad(d)))

        for idx, leaky_conv in enumerate(self.data_convs):
            res = leaky_conv(u)
            u = self.dat_batch_norm[idx](self.leaky_relu(u + res))
            down_sample = (idx+1) % (self._data_layers) == 0
            if down_sample:
                u = self.avg_pool(u)
        return u

    def latent_space_forward(self, u: latent_variable, v: latent_variable):
        ''' Fixed point operator on latent space (when v is fixed)

            R(u,v) is used in fixed point iteration of FPN to find u*
            satisfying
            u* = R(u*, v). To make R be a contraction in u, we estimate a
            Lipschitz constant and normalize updates using this.
        '''
        if self.architecture == 'Jacobian':
            R_uv = v + self.gamma * u
            for idx, leaky_conv in enumerate(self.latent_convs):
                res = leaky_conv(R_uv)
                R_uv = self.leaky_relu(R_uv + res)
            return R_uv
        else:
            R_uv = v + self.gamma * u
            for idx, leaky_conv in enumerate(self.latent_convs):
                res = leaky_conv(R_uv)
                R_uv = self.lat_batch_norm[idx](self.leaky_relu(R_uv + res))
            return R_uv

    def map_latent_to_inference(self, u: latent_variable) -> classification:
        ''' Transform feature vectors into a classification

            This is the final step of FPN, which flattens and
            then applies affine mappings to input. Operations do *not* need to
            be 1-Lipschitz.
        '''
        y = self.drop_outS(self.leaky_relu(self.conv_y(u)))
        y = y.view(u.size()[0], -1)
        class_label = self.label_fc(y)
        return class_label

    def forward(self, d: image, eps=1.0e-3, max_depth=100,
                depth_warning=False) -> classification:
        ''' FPN forward prop

            With gradients detached, find fixed point. During forward
            iteration,
            u is updated via R(u,Q(d)) and Lipschitz constant estimates are
            refined. Gradient are attached performing one final step.
        '''
        exp_name = 'Explicit'
        exp_unaug = 'Explicit_Unaugmented'
        if self.architecture == exp_name or self.architecture == exp_unaug:
            return forward_explicit(self, d, eps=eps, max_depth=max_depth,
                                    depth_warning=False)
        else:
            return forward_implicit(self, d, eps=eps, max_depth=max_depth,
                                    depth_warning=False)

    def normalize_lip_const(self, u: latent_variable, v: latent_variable):
        return normalize_lip_const(self, u, v)
