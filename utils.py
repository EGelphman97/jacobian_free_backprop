import torch
from prettytable import PrettyTable
from time import sleep
import time
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from BatchCG import cg_batch
from torch.cuda import memory_allocated, max_memory_allocated, utilization
from tracemalloc import start, get_traced_memory
from torch.profiler import profile, record_function, ProfilerActivity

def get_stats(net, test_loader, criterion, num_classes: int, eps: float,
              max_depth: int):
    test_loss = 0
    num_correct_labels = 0

    with torch.no_grad():
        for d_test, labels in test_loader:
            labels = labels.to(net.device())
            d_test = d_test.to(net.device())
            batch_size = d_test.shape[0]
            if net.name() == "MNIST_FCN":
                d_test = d_test.view(d_test.size()[0], 784).to(net.device())

            ut = torch.zeros((d_test.size()[0], num_classes))
            ut = ut.to(net.device())
            for i in range(d_test.size()[0]):
                ut[i, labels[i].cpu().numpy()] = 1.0

            y = net(d_test, eps=eps, max_depth=max_depth)

            if str(criterion) == "MSELoss()":
                batch_loss = criterion(y.double(), ut.double()).item()
                test_loss += batch_size * batch_loss
            elif str(criterion) == "CrossEntropyLoss()":
                test_loss += batch_size * criterion(y, labels).item()
            else:
                print("Error: Invalid Loss Function")

            pred = y.argmax(dim=1, keepdim=True)
            num_correct_labels += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * num_correct_labels/len(test_loader.dataset)
    return test_loss, test_acc, num_correct_labels


def model_params(net):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['TOTAL', num_params])
    return table


def train_class_net(net, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, criterion,
                    num_classes, eps, max_depth, device, save_dir='./'):

    fmt = '[{:3d}/{:3d}]: train - ({:6.2f}%, {:6.2e}), test - ({:6.2f}%, '
    fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

    depth_ave = 0.0
    pct_ls_ave = 0.0 #percentage of anderson acceleration iterations that require least squares solve on average per epoch
    train_acc = 0.0
    best_test_acc = 0.0

    total_time = 0.0
    time_hist = []
    test_loss_hist = []
    test_acc_hist = []
    train_loss_hist = []
    train_acc_hist = []
    avg_depth_hist = []
    depth_max_hist = []
    pct_ls_hist = []
    util_hist = []

    print(net)
    print(model_params(net))
    print('\nTraining Fixed Point Network')

    mem_epoch_hist = []
    peak_mem = 0.0
    if device == "cpu":#Start memory allocation measurement using tracemalloc
        start()

    for epoch in range(max_epochs):
        sleep(0.5)  # slows progress bar so it won't print on multiple lines
        loss_ave = 0.0
        epoch_start_time = time.time()
        tot = len(train_loader)
        depth_max = -1#Maximum depth of fixed point network in current epoch
        util = 0.0
        mem_epoch = 0.0
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))
            count = 0
            for _, (d, labels) in enumerate(train_loader):
                labels = labels.to(net.device())
                d = d.to(net.device())
                batch_size = d.shape[0]
                if net.name() == "MNIST_FCN":
                    d = d.view(d.size()[0], 784).to(net.device())
                # -------------------------------------------------------------
                # Apply network to get fixed point and then backprop
                # -------------------------------------------------------------
                optimizer.zero_grad()
                y = net(d, eps=eps, max_depth=max_depth)

                depth_ave = 0.99 * depth_ave + 0.01 * net.depth
                pct_ls_ave = 0.0#0.99 * pct_ls_ave + 0.01 * (float(net.num_ls)/float(net.depth))
                if net.depth > depth_max:
                    depth_max = net.depth
                output = None
                if str(criterion) == "MSELoss()":
                    ut = torch.zeros((batch_size, num_classes))
                    ut = ut.to(net.device())
                    for i in range(batch_size):
                        ut[i, labels[i].cpu().numpy()] = 1.0
                    output = criterion(y.double(), ut.double())
                elif str(criterion) == "CrossEntropyLoss()":
                    output = criterion(y, labels)
                else:
                    print("Error: Invalid Loss Function")
                loss_val = output.detach().cpu().numpy() * batch_size
                loss_ave += loss_val
                output.backward()
                optimizer.step()
                # -------------------------------------------------------------
                # Output training stats
                # -------------------------------------------------------------
                pred = y.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                train_acc = 0.99 * train_acc + 1.00 * correct / batch_size
                tepoch.update(1)
                tepoch.set_postfix(train_loss="{:5.2e}".format(loss_val
                                   / batch_size),
                                   train_acc="{:5.2f}%".format(train_acc),
                                   depth="{:5.1f}".format(net.depth))
                util += utilization(device)
                count += 1
                if device == "cuda":
                    mem_epoch += memory_allocated(device)

        #  divide by total number of training samples
        loss_ave = loss_ave / len(train_loader.dataset)

        test_loss, test_acc, correct = get_stats(net,
                                                 test_loader,
                                                 criterion,
                                                 num_classes,
                                                 eps,
                                                 max_depth)

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(loss_ave)
        train_acc_hist.append(train_acc)

        if device == "cuda":
            mem_epoch_hist.append(mem_epoch/count)
            util_hist.append(util/count)
        elif device == "cpu":
            mem_epoch += get_traced_memory()[0]#Get current, not peak
        avg_depth_hist.append(depth_ave)
        depth_max_hist.append(depth_max)
        pct_ls_hist.append(pct_ls_ave)

        epoch_end_time = time.time()
        time_epoch = epoch_end_time - epoch_start_time
        time_hist.append(time_epoch)
        total_time += time_epoch

        print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave,
                         test_acc, test_loss, depth_ave,
                         optimizer.param_groups[0]['lr'],
                         time_epoch))
        # ---------------------------------------------------------------------
        # Save weights
        # ---------------------------------------------------------------------
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler
            }
            file_name = save_dir + net.name() + '_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)
        # ---------------------------------------------------------------------
        # Save history at last epoch
        # ---------------------------------------------------------------------
        if epoch+1 == max_epochs:
            #mem_epoch /= max_epochs
            if device == "cuda":
                peak_mem = max_memory_allocated(device)
            elif device == "cpu":
                peak_mem = get_traced_memory()[1]
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'train_loss_hist': train_loss_hist,
                'train_acc_hist': train_acc_hist,
                'lr_scheduler': lr_scheduler,
                'time_hist': time_hist,
                'eps': eps,
                'avg_mem': np.mean(np.array(mem_epoch_hist)),
                'peak_mem': peak_mem,
                'avg_depth_hist': avg_depth_hist,
                'depth_max_hist': depth_max_hist,
                'avg_pct_ls': np.mean(np.array(pct_ls_hist)),
                'util_hist': util_hist,
            }
            file_name = save_dir + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)

        lr_scheduler.step()
        epoch_start_time = time.time()
    return net

def train_class_net_prof(net, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, criterion,
                    num_classes, eps, max_depth, device, save_dir='./'):

    fmt = '[{:3d}/{:3d}]: train - ({:6.2f}%, {:6.2e}), test - ({:6.2f}%, '
    fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

    depth_ave = 0.0
    train_acc = 0.0
    best_test_acc = 0.0

    total_time = 0.0
    time_hist = []
    test_loss_hist = []
    test_acc_hist = []
    train_loss_hist = []
    train_acc_hist = []
    avg_depth_hist = []
    depth_max_hist = []

    print(net)
    print(model_params(net))
    print('\nTraining Fixed Point Network')

    mem_epoch = 0.0
    peak_mem = 0.0

    for epoch in range(max_epochs):
        sleep(0.3)  # slows progress bar so it won't print on multiple lines
        loss_ave = 0.0
        epoch_start_time = time.time()
        tot = len(train_loader)
        depth_max = -1#Maximum depth of fixed point network in current epoch
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
            with record_function("forward_pass_and_backprop"):
                with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

                    tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))

                    for _, (d, labels) in enumerate(train_loader):
                        labels = labels.to(net.device())
                        d = d.to(net.device())
                        batch_size = d.shape[0]
                        if net.name() == "MNIST_FCN":
                            d = d.view(d.size()[0], 784).to(net.device())
                        # -------------------------------------------------------------
                        # Apply network to get fixed point and then backprop
                        # -------------------------------------------------------------
                        optimizer.zero_grad()
                        y = net(d, eps=eps, max_depth=max_depth)

                        depth_ave = 0.99 * depth_ave + 0.01 * net.depth
                        if net.depth > depth_max:
                            depth_max = net.depth
                        output = None
                        if str(criterion) == "MSELoss()":
                            ut = torch.zeros((batch_size, num_classes))
                            ut = ut.to(net.device())
                            for i in range(batch_size):
                                ut[i, labels[i].cpu().numpy()] = 1.0
                            output = criterion(y.double(), ut.double())
                        elif str(criterion) == "CrossEntropyLoss()":
                            output = criterion(y, labels)
                        else:
                            print("Error: Invalid Loss Function")
                        loss_val = output.detach().cpu().numpy() * batch_size
                        loss_ave += loss_val
                        output.backward()
                        optimizer.step()
                        # -------------------------------------------------------------
                        # Output training stats
                        # -------------------------------------------------------------
                        pred = y.argmax(dim=1, keepdim=True)
                        correct = pred.eq(labels.view_as(pred)).sum().item()
                        train_acc = 0.99 * train_acc + 1.00 * correct / batch_size
                        tepoch.update(1)
                        tepoch.set_postfix(train_loss="{:5.2e}".format(loss_val
                                        / batch_size),
                                        train_acc="{:5.2f}%".format(train_acc),
                                        depth="{:5.1f}".format(net.depth))

                #  divide by total number of training samples
                loss_ave = loss_ave / len(train_loader.dataset)

                test_loss, test_acc, correct = get_stats(net,
                                                        test_loader,
                                                        criterion,
                                                        num_classes,
                                                        eps,
                                                        max_depth)

        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=15))
        prof.export_chrome_trace(save_dir + "exec_trace.json")

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(loss_ave)
        train_acc_hist.append(train_acc)
        avg_depth_hist.append(depth_ave)
        depth_max_hist.append(depth_max)

        epoch_end_time = time.time()
        time_epoch = epoch_end_time - epoch_start_time
        time_hist.append(time_epoch)
        total_time += time_epoch

        print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave,
                         test_acc, test_loss, depth_ave,
                         optimizer.param_groups[0]['lr'],
                         time_epoch))
        # ---------------------------------------------------------------------
        # Save weights
        # ---------------------------------------------------------------------
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler
            }
            file_name = save_dir + net.name() + '_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)
        # ---------------------------------------------------------------------
        # Save history at last epoch
        # ---------------------------------------------------------------------
        if epoch+1 == max_epochs:
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'train_loss_hist': train_loss_hist,
                'train_acc_hist': train_acc_hist,
                'lr_scheduler': lr_scheduler,
                'time_hist': time_hist,
                'eps': eps,
                'avg_mem': mem_epoch,
                'peak_mem': peak_mem,
                'avg_depth_hist': avg_depth_hist,
                'depth_max_hist': depth_max_hist,
            }
            file_name = save_dir + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)

        lr_scheduler.step()
        epoch_start_time = time.time()
    return net

def mnist_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    train_loader = train_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('data',
                                       train=True,
                                       download=True,
                                       transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                             (0.3081,))
                                       ])),
                        batch_size=train_batch_size,
                        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('data',
                                       train=False,
                                       transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                             (0.3081,))
                                        ])),
                        batch_size=test_batch_size,
                        shuffle=False)
    return train_loader, test_loader


def svhn_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                     std=[0.1980, 0.2010, 0.1970])
    train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root='data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ]),
            ),
            batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root='data', split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def cifar_loaders(train_batch_size, test_batch_size=None, augment=True):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if augment:
        transforms_list = [transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           normalize,
                           transforms.RandomCrop(32, 2, fill=0.449),
                           transforms.RandomErasing(p=0.95, scale=(0.1, 0.25),
                                                    ratio=(0.2, 5.0),
                                                    value=[0.485, 0.456,
                                                           0.406])
                           ]
    else:
        transforms_list = [transforms.ToTensor(), normalize]

    trans_comp = transforms.Compose(transforms_list)
    train_dataset = datasets.CIFAR10('data',
                                     train=True,
                                     download=True,
                                     transform=trans_comp)
    test_dataset = datasets.CIFAR10('data',
                                    train=False,
                                    transform=trans_comp)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False, pin_memory=True)
    return train_loader, test_loader


# ------------------------------------------------------------------------------
# Jacobian-based functions
# ------------------------------------------------------------------------------
def compute_fixed_point(T, Qd, max_depth, device, eps=1e-4, use_anderson=False):

    depth = 0.0
    num_ls = 0
    u = torch.zeros(Qd.shape, device=T.device())
    u_prev = np.Inf * torch.ones(u.shape, device=T.device())

    # approximately normalize weights by lipschitz constant before
    # computing fixed point
    T.normalize_lip_const(u, Qd)
    with torch.no_grad():
        if not use_anderson:
            all_samp_conv = False
            while not all_samp_conv and depth < max_depth:
                u_prev = u.clone()
                u = T.latent_space_forward(u, Qd)
                depth += 1.0
                all_samp_conv = torch.max(torch.norm(u - u_prev, dim=1)) <= eps
        else:
            u, u_prev, num_itr, num_ls_aa = anderson(T, u, Qd, tol=eps, max_iters=max_depth, beta=1.5)
            depth += num_itr
            num_ls += num_ls_aa

    return u.detach(), depth, num_ls


def train_Jacobian_based_net(net, max_epochs, lr_scheduler, train_loader,
                             test_loader, optimizer, criterion,
                             num_classes, eps, max_depth, save_dir='./',
                             JTJ_shift=0.0):

    avg_time = 0.0
    total_time = 0.0
    time_hist = []
    n_Umatvecs = []
    max_iter_cg = max_depth
    tol_cg = eps

    depth_ave = 0.0
    pct_ls_ave = 0.0
    best_test_acc = 0.0
    train_acc = 0.0

    test_loss_hist = []   # test loss history array
    test_acc_hist = []    # test accuracy history array
    depth_test_hist = []  # test depths history array
    train_loss_hist = []  # train loss history array
    train_acc_hist = []   # train accuracy history array
    pct_lst_hist = []     # percentage of anderson acceleration iterations that require least squares solve history

    fmt = '[{:4d}/{:4d}]: train acc = {:5.2f}% | train_loss = {:7.3e} | '
    fmt += ' test acc = {:5.2f}% | test loss = {:7.3e} | '
    fmt += 'depth = {:5.1f} | lr = {:5.1e} | time = {:4.1f} sec | n_Umatvecs '
    fmt += '= {:4d} | cg = {:7.3e}'
    print(net)                 # display Tnet configuration
    print(model_params(net))   # display Tnet parameters
    print('\nTraining Jacobian-based Network')

    mem_epoch = 0.0
    peak_mem = 0.0
    if net.device == "cpu":#Start memory allocation measurement using tracemalloc
        start()

    for epoch in range(max_epochs):

        sleep(0.5)  # slows progress bar so it won't print on multiple lines
        tot = len(train_loader)
        temp_n_Umatvecs = 0  # XXX - return and explain
        cg_iters = 0
        start_time_epoch = time.time()
        temp_max_depth = 0
        loss_ave = 0.0
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))

            for idx, (d, labels) in enumerate(train_loader):
                labels = labels.to(net.device())
                d = d.to(net.device())

                # --------------------------------------------------------------
                # Find Fixed Point
                # --------------------------------------------------------------
                train_batch_size = d.shape[0]  # redefine if batch size changes
                # u0 = torch.zeros((train_batch_size, lat_dim)).to(device)
                with torch.no_grad():
                    Qd = net.data_space_forward(d)
                    u, depth, num_ls = compute_fixed_point(net, Qd, max_depth,
                                                   net.device(), eps=eps, use_anderson=net.use_anderson)

                    depth_ave = 0.99 * depth_ave + 0.01 * depth
                    pct_ls_ave = 0.99 * depth_ave + 0.01 * (float(num_ls)/float(depth))
                    temp_max_depth = max(depth, temp_max_depth)

                # -------------------------------------------------------------
                # Jacobian_Based Backprop
                # -------------------------------------------------------------
                net.train()
                optimizer.zero_grad()  # Initialize gradient to zero

                # compute output for backprop
                u.requires_grad = True
                Qd = net.data_space_forward(d)

                Ru = net.latent_space_forward(u, Qd)
                S_Ru = net.map_latent_to_inference(Ru)
                loss = criterion(S_Ru, labels)
                train_loss = loss.detach().cpu().numpy() * train_batch_size
                loss_ave += train_loss

                # -------------------------------------------------------------
                # compute rhs = dldu * J^T
                # -------------------------------------------------------------

                # dldu = dl/dS * dS/du
                dldu = torch.autograd.grad(outputs=loss, inputs=Ru,
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                # compute dldu * J
                dldu_dRdu = torch.autograd.grad(outputs=Ru, inputs=u,
                                                grad_outputs=dldu,
                                                retain_graph=True,
                                                create_graph=True,
                                                only_inputs=True)[0]
                dldu_J = dldu - dldu_dRdu

                # autograd trick: d(v*J)/v * v = v * J^T
                dldu_JT = torch.autograd.grad(outputs=dldu_J, inputs=dldu,
                                              grad_outputs=dldu,
                                              retain_graph=True,
                                              create_graph=True,
                                              only_inputs=True)[0]
                rhs = dldu_JT

                rhs = rhs.detach()
                # vectorize channels (when R is a CNN)
                rhs = rhs.view(train_batch_size, -1)
                # CG requires it to have dims: n_samples x n_features x n_rh
                rhs = rhs.unsqueeze(2)  # unsqueeze for number of rhs.

                # -------------------------------------------------------------
                # Define JJT matvec function
                # -------------------------------------------------------------

                def v_JJT_matvec(v, u=u, Ru=Ru):
                    # inputs:
                    # v = vector to be multiplied by JJT
                    # u = fixed point vector u (requires grad)
                    # Ru = R applied to u (requires grad)

                    # assumes one rhs:
                    # x (n_samples, n_dim, n_rhs) -> (n_samples, n_dim)

                    v = v.squeeze(2)      # squeeze number of RHS
                    v = v.view(Ru.shape)  # reshape to filter space
                    v.requires_grad = True

                    # compute v*J = v*(I - dRdu)
                    v_dRdu = torch.autograd.grad(outputs=Ru, inputs=u,
                                                 grad_outputs=v,
                                                 retain_graph=True,
                                                 create_graph=True,
                                                 only_inputs=True)[0]
                    v_J = v - v_dRdu

                    # compute v_JJT
                    v_JJT = torch.autograd.grad(outputs=v_J, inputs=v,
                                                grad_outputs=v_J,
                                                retain_graph=True,
                                                create_graph=True,
                                                only_inputs=True)[0]

                    v = v.detach()
                    v_J = v_J.detach()
                    Amv = v_JJT.detach()
                    Amv = Amv.view(Ru.shape[0], -1)
                    Amv = Amv.unsqueeze(2).detach()
                    return Amv

                normal_eq_sol, info = cg_batch(v_JJT_matvec, rhs, M_bmm=None,
                                               X0=None, rtol=0, atol=tol_cg,
                                               maxiter=max_iter_cg,
                                               verbose=False)
                # JTJinv_v has size (batch_size x n_hidden_features)
                # n_rhs is squeezed
                normal_eq_sol = normal_eq_sol.squeeze(2)
                normal_eq_sol = normal_eq_sol.view(Ru.shape)

                temp_n_Umatvecs += info['niter'] * train_batch_size
                cg_iters += info['niter']

                if info['optimal']:
                    # avoid updating "bad batches", update when CG converges

                    # compute dTdtheta
                    # Ru = Ru.view(train_batch_size, -1)
                    # reshape in case Ru is a CNN
                    # computes
                    # v_JJTinv_dRdTheta = dSdu * dldS * Jinv * dRdTheta
                    u.requires_grad = False
                    Ru.backward(normal_eq_sol)

                    S_Ru = net.map_latent_to_inference(Ru.detach())
                    loss = criterion(S_Ru, labels)
                    loss.backward()
                    u.requires_grad = False
                    optimizer.step()

                # -------------------------------------------------------------
                # Output training stats
                # -------------------------------------------------------------
                pred = S_Ru.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                train_acc = 0.99 * train_acc + 1.0 * correct / train_batch_size
                tepoch.update(1)
                tepoch.set_postfix(train_loss="{:5.2e}".format(train_loss
                                   / train_batch_size),
                                   train_acc="{:5.2f}%".format(train_acc),
                                   depth="{:5.1f}".format(temp_max_depth),
                                   cgiters="{:5.1f}".format(info['niter']))
        loss_ave /= len(train_loader.dataset)

        # update optimization scheduler
        lr_scheduler.step()

        # compute test loss and accuracy
        test_loss, test_acc, correct = get_stats(net, test_loader, criterion,
                                                 10, eps, max_depth)
        # test_loss, test_acc, correct = get_stats_Jacobian(net, test_loader,
        # criterion, eps, max_depth)

        #Memory
        if net.device == "cuda":
            mem_epoch += memory_allocated(net.device)
        elif net.device == "cpu":
            mem_epoch += get_traced_memory()[0]#Get current, not peak

        end_time_epoch = time.time()
        time_epoch = end_time_epoch - start_time_epoch

        # ---------------------------------------------------------------------
        # Compute costs and statistics
        # ---------------------------------------------------------------------
        time_hist.append(time_epoch)
        total_time += time_epoch
        avg_time /= total_time/(epoch+1)
        n_Umatvecs.append(temp_n_Umatvecs)

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(loss_ave)
        train_acc_hist.append(train_acc)
        depth_test_hist.append(net.depth)
        pct_lst_hist.append(pct_ls_ave)

        # ---------------------------------------------------------------------
        # Print outputs to console
        # ---------------------------------------------------------------------

        print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave,
                         test_acc, test_loss, temp_max_depth,
                         optimizer.param_groups[0]['lr'],
                         time_epoch, temp_n_Umatvecs, cg_iters))

        # ---------------------------------------------------------------------
        # Save weights
        # ---------------------------------------------------------------------
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler
            }
            file_name = save_dir + net.name() + '_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)

        # ---------------------------------------------------------------------
        # Save history at last epoch
        # ---------------------------------------------------------------------
        if epoch+1 == max_epochs:
            mem_epoch /= max_epochs
            if net.device == "cuda":
                peak_mem = max_memory_allocated(net.device)
            elif net.device == "cpu":
                peak_mem = get_traced_memory()[1]
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'train_loss_hist': train_loss_hist,
                'train_acc_hist': train_acc_hist,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler,
                'avg_time': avg_time,
                'n_Umatvecs': n_Umatvecs,
                'time_hist': time_hist,
                'tol_cg': tol_cg,
                'eps': eps,
                'net_state_dict': net.state_dict(),
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'depth_test_hist': depth_test_hist,
                'avg_mem': mem_epoch,
                'peak_mem': peak_mem,
                'avg_pct_ls': np.mean(np.array(pct_lst_hist)),
            }
            file_name = save_dir + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)

    return net


def train_Neumann_FPN_net(net, max_epochs, lr_scheduler, train_loader,
                          test_loader, optimizer, criterion,
                          num_classes, eps, max_depth, save_dir='./',
                          neumann_order=0):

    avg_time = 0.0
    total_time = 0.0
    time_hist = []
    n_Umatvecs = []

    depth_ave = 0.0
    best_test_acc = 0.0
    train_acc = 0.0

    test_loss_hist = []   # test loss history array
    test_acc_hist = []    # test accuracy history array
    depth_test_hist = []  # test depths history array
    train_loss_hist = []  # train loss history array
    train_acc_hist = []   # train accuracy history array

    fmt = '[{:4d}/{:4d}]: train acc = {:5.2f}% | train_loss = {:7.3e} | '
    fmt += ' test acc = {:5.2f}% | test loss = {:7.3e} | '
    fmt += 'depth = {:5.1f} | lr = {:5.1e} | time = {:4.1f} sec | '
    fmt += 'n_Umatvecs = {:4d}'
    print(net)                 # display Tnet configuration
    print(model_params(net))   # display Tnet parameters
    print('\nTraining Neumann-based Network')

    for epoch in range(max_epochs):

        sleep(0.5)  # slows progress bar so it won't print on multiple lines
        tot = len(train_loader)
        temp_n_Umatvecs = 0
        cg_iters = 0
        start_time_epoch = time.time()
        temp_max_depth = 0
        loss_ave = 0.0
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))

            for idx, (d, labels) in enumerate(train_loader):
                labels = labels.to(net.device())
                d = d.to(net.device())

                # -------------------------------------------------------------
                # Find Fixed Point
                # -------------------------------------------------------------
                train_batch_size = d.shape[0]  # redefine if batch size changes

                with torch.no_grad():
                    Qd = net.data_space_forward(d)
                    u, depth = compute_fixed_point(net, Qd, max_depth,
                                                   net.device(), eps=eps)

                    depth_ave = 0.99 * depth_ave + 0.01 * net.depth

                    temp_max_depth = max(depth, temp_max_depth)

                # -------------------------------------------------------------
                # Jacobian_Based Backprop
                # -------------------------------------------------------------
                net.train()
                optimizer.zero_grad()  # Initialize gradient to zero

                # compute output for backprop
                u.requires_grad = True
                Qd = net.data_space_forward(d)

                Ru = net.latent_space_forward(u, Qd)
                S_Ru = net.map_latent_to_inference(Ru)
                loss = criterion(S_Ru, labels)
                train_loss = loss.detach().cpu().numpy() * train_batch_size
                loss_ave += train_loss

                dldS_dSdu = torch.autograd.grad(outputs=loss, inputs=Ru,
                                                retain_graph=True,
                                                create_graph=True,
                                                only_inputs=True)[0]
                dldS_dSdu = dldS_dSdu.detach()  # dldu = dS/du * dl/dS

                dldS_dSdu_Jinv_approx = dldS_dSdu.clone().detach()
                dldS_dSdu_dRdu_k = dldS_dSdu.clone().detach()

                # Approximate Jacobian inverse with Neumann series expansion
                # up to neumann_order terms
                for i in range(1, neumann_order+1):

                    dldS_dSdu_dRdu_k.requires_grad = True

                    # compute dldu_dRdu_k * dRdu = dldu_dRdu_k+1
                    dldS_dSdu_dRdu_kplus1 = torch.autograd.grad(
                                            outputs=Ru,
                                            inputs=u,
                                            grad_outputs=dldS_dSdu_dRdu_k,
                                            retain_graph=True,
                                            create_graph=True,
                                            only_inputs=True)[0]

                    dldS_dSdu_Jinv_approx = dldS_dSdu_Jinv_approx + dldS_dSdu_dRdu_kplus1.detach()

                    dldS_dSdu_dRdu_k = dldS_dSdu_dRdu_kplus1.detach()

                    temp_n_Umatvecs += int(neumann_order*(neumann_order+1)/2)
                Ru.backward(dldS_dSdu_Jinv_approx)

                S_Ru = net.map_latent_to_inference(Ru.detach())
                loss = criterion(S_Ru, labels)
                loss.backward()

                u.requires_grad = False

                # update net parameters
                optimizer.step()

                # -------------------------------------------------------------
                # Output training stats
                # -------------------------------------------------------------
                pred = S_Ru.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                train_acc = 0.99 * train_acc + 1.0 * correct / train_batch_size
                tepoch.update(1)
                tepoch.set_postfix(train_loss="{:5.2e}".format(train_loss
                                   / train_batch_size),
                                   train_acc="{:5.2f}%".format(train_acc),
                                   depth="{:5.1f}".format(temp_max_depth))
        loss_ave /= len(train_loader.dataset)

        # update optimization scheduler
        lr_scheduler.step()

        # compute test loss and accuracy
        test_loss, test_acc, correct = get_stats(net, test_loader, criterion,
                                                 10, eps, max_depth)

        end_time_epoch = time.time()
        time_epoch = end_time_epoch - start_time_epoch

        # ---------------------------------------------------------------------
        # Compute costs and statistics
        # ---------------------------------------------------------------------
        time_hist.append(time_epoch)
        total_time += time_epoch
        avg_time /= total_time/(epoch+1)
        n_Umatvecs.append(temp_n_Umatvecs)

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(loss_ave)
        train_acc_hist.append(train_acc)
        depth_test_hist.append(net.depth)

        # ---------------------------------------------------------------------
        # Print outputs to console
        # ---------------------------------------------------------------------

        print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave,
                         test_acc, test_loss, temp_max_depth,
                         optimizer.param_groups[0]['lr'],
                         time_epoch, temp_n_Umatvecs, cg_iters))

        # ---------------------------------------------------------------------
        # Save weights
        # ---------------------------------------------------------------------
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler
            }
            file_name = save_dir + net.name() + '_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)

        # ---------------------------------------------------------------------
        # Save history at last epoch
        # ---------------------------------------------------------------------
        if epoch+1 == max_epochs:
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'train_loss_hist': train_loss_hist,
                'train_acc_hist': train_acc_hist,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler,
                'avg_time': avg_time,
                'n_Umatvecs': n_Umatvecs,
                'time_hist': time_hist,
                'eps': eps,
                'net_state_dict': net.state_dict(),
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'depth_test_hist': depth_test_hist
            }
            file_name = save_dir + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)
    return net

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
            Fixed point of T_eval, value of u after previous iteration, number 
            of iterations, and number of least sqares solves
        """
        batch_sz, d, h, w = u0.shape
        u_hist = torch.zeros(batch_sz, m, d*h*w, dtype=u0.dtype, device=u0.device)
        T_hist = torch.zeros(batch_sz, m, d*h*w, dtype=u0.dtype, device=u0.device)
        u_hist[:,0] = u0.view(batch_sz, -1)
        T_hist[:,0] = net.latent_space_forward(u0, Qd).view(batch_sz,-1)
        u_hist[:,1] = T_hist[:,0]
        T_hist[:,1] = net.latent_space_forward(T_hist[:,0].view_as(u0), Qd).view(batch_sz,-1)
        H = torch.zeros(batch_sz, m+1, m+1, dtype=u0.dtype, device=u0.device)
        H[:,0,1:] = 1.0
        H[:,1:,0] = 1.0
        Batch_RHS = torch.zeros(batch_sz, m+1, 1, dtype=u0.dtype, device=u0.device)
        Batch_RHS[:,0] = 1.0 

        #res_k = ((T_hist[:,0] - u_hist[:,0]).norm().item()) / (1.0e-9 + T_hist[:,0].norm().item())
        k = 1
        res_k = ((T_hist[:,k%m] - u_hist[:,k%m]).norm().item()) / (1.0e-9 + T_hist[:,k%m].norm().item())
        k += 1
        lstsq_solves = 0 
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
                lstsq_solves += 1
            #Update data structures
            u_hist[:,k%m] = (1.0-beta)*((alpha[:,None]@u_hist[:,:M])[:,0]) + beta*((alpha[:,None]@T_hist[:,:M])[:,0])
            T_hist[:,k%m] = net.latent_space_forward(u_hist[:,k%m].view_as(u0), Qd).view(batch_sz, -1)
            res_k = ((T_hist[:,k%m] - u_hist[:,k%m]).norm().item()) / (1.0e-9 + T_hist[:,k%m].norm().item())
            k += 1

        return u_hist[:,k%m].view_as(u0), u_hist[:,(k-1)%m].view_as(u0), k, lstsq_solves
