import os
import argparse
import logging
import time

import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from numpy.linalg import svd
from torch.autograd import Variable, grad
from torch.autograd.functional import jacobian
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from Jacobian import extend, JacobianMode


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=5)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--ortho_decay', '--od', default=1e-2, type=float,
                    help='ortho weight decay')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')

args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

features_in_hook = []
features_out_hook = []


def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None


def jacobian_temp(inputs, outputs):
    #inputs = Variable(inputs).to(device).requires_grad_()
    #allow_unused=True
    return torch.stack(
        [grad([outputs[:, i].sum()], [inputs], retain_graph=True, create_graph=True)[0] for i in
         range(outputs.size(1))], dim=-1)


def adjust_weight_decay_rate(optimizer, epoch):
    w_d = args.weight_decay

    if epoch > 20:
        w_d = 5e-4
    elif epoch > 10:
        w_d = 1e-6

    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = w_d


def adjust_ortho_decay_rate(epoch):
    o_d = args.ortho_decay

    if epoch > 120:
        o_d = 0.0
    elif epoch > 70:
        o_d = 1e-6 * o_d
    elif epoch > 50:
        o_d = 1e-4 * o_d
    elif epoch > 20:
        o_d = 1e-3 * o_d

    return o_d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


def l2_reg_ortho(mdl):
    l2_reg = None
    # for W in mdl.nparameters():
    for name, W in mdl.named_parameters():
        if name[0] == '7' and name[-1] == 't':
            if W.ndimension() < 2:
                continue
            else:
                # if name[0] == '7':
                cols = W[0].numel()
                rows = W.shape[0]
                w1 = W.view(-1, cols)
                wt = torch.transpose(w1, 0, 1)
                if (rows > cols):
                    m = torch.matmul(wt, w1)
                    ident = Variable(torch.eye(cols, cols), requires_grad=True)
                else:
                    m = torch.matmul(w1, wt)
                    ident = Variable(torch.eye(rows, rows), requires_grad=True)

                ident = ident.cuda()
                w_tmp = (m - ident)
                b_k = Variable(torch.rand(w_tmp.shape[1], 1))
                b_k = b_k.cuda()

                v1 = torch.matmul(w_tmp, b_k)
                norm1 = torch.norm(v1, 2)
                v2 = torch.div(v1, norm1)
                v3 = torch.matmul(w_tmp, v2)

                if l2_reg is None:
                    l2_reg = (torch.norm(v3, 2)) ** 2
                else:
                    l2_reg = l2_reg + (torch.norm(v3, 2)) ** 2
    return l2_reg


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    lastfunc = ODEBlock(ODEfunc(64)).to(device)
    #extend(model,(1, 28, 28))
    parm = {}
    for name, parameters in model.named_parameters():
        if name == '7.odefunc.conv1._layer.weight':
            print(name, ':', parameters.size())

        # parm[name]=parameters.detach().numpy()
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()
    ortho_decay = args.ortho_decay
    weight_decay = args.weight_decay
    #Jx = []
    Jaco = []
    sv = []
    # net = ODEBlock()
    print("batches_per_epoch: ", batches_per_epoch)
    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)
        odecay = adjust_ortho_decay_rate(itr + 1)
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        #Jx = x
        #print('Jx:', Jx.shape)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        '''
        with JacobianMode(model):
            logits = model(x)
            if itr % batches_per_epoch == 0:
                temp = logits.sum().backward()
                jac = model.jacobian()
                Jaco.append(jac)
                logger.info('Jaco append')
        #logits = x
        '''
        
        '''
        for i in range(len(model)):
            logits = model[i](logits)
            #print(itr % batches_per_epoch)
            if itr % batches_per_epoch == 0:
               if i == 6:
                  #print(logit.shape)
                  #extend(model[i],(128,))
                  with JacobianMode(model[i]):
                      out = model[i](x)
                      out.sum().backward()
                      jac = net.jacobian()
                  print('jac append')
                  Jx.append(jac)
                  print('Jac.len:',len(Jx))
                          
               if i == 12:
                  print('Jy append')
                  Jy.append(logits)
        '''

        #print('len Jy Jx', len(Jy),'  ',len(Jx))
            #print('layar ',i,' :',logits.shape)
        '''
            if i == 6:
               print("layer 6:",logits.shape)
            if i == 7:
               print("layer 7:",logits.shape)
        '''
        # loss = criterion(logits, y)
        oloss = l2_reg_ortho(model)
        oloss = odecay * oloss
        loss = criterion(logits, y)
        loss = loss + oloss
        '''
        #Jy = logits
        if len(Jx) > 0 and itr % batches_per_epoch == 0:
            print('Jx.shape',Jx[-1].shape)
            
            #print('Jy.shape',Jy[-1].shape)
            J = jacobian(lastfunc,Jx[-1])
            sv.append(svd(J.detach().cpu().numpy(), compute_uv=False))
            print('sv.len: ', len(sv))
        '''
        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        if itr % batches_per_epoch == 0:
            Jac = []
            for o in logits.view(-1):
                #model.zero_grad()
                grad = []
                #o.backward(retain_graph=True)
                for param in model.parameters():
                    grad.append(param.grad.reshape(-1))
                Jac.append(torch.cat(grad))
            Jac = torch.stack(Jac)
            print('Jac.shape',Jac.shape)
            Jaco.append(Jac)        
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            '''
            for (name, module) in model.named_modules():
                if name in 'odefunc.':
                    module.register_forward_hook(hook=hook)
                    if len(features_in_hook) > 0:
                        print("shape for in", features_in_hook[-1].shape)
            '''
            #jaco = jacobian(x, logits)
            sv.append(svd(Jaco[-1].cpu().numpy(), compute_uv=False))
            print('sv.len:',len(sv))
            print('sv.shape:',sv[-1].shape)
            with torch.no_grad():
                '''
                for name, param in model.named_parameters():
                    if name == '7.odefunc.conv1._layer.weight':
                        sv.append(svd(param[1].detach().cpu().numpy(), compute_uv=False))
                        print('sv:', len(sv))
                '''
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )
    '''            
    for i in range(len(Jx)):
        Jx[i] = Jx[i].detach().cpu().numpy()
    np.savez('Jx',Jx) 
    '''
