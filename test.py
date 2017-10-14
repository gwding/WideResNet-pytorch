"""Code copied and adapted from
https://github.com/xternalz/WideResNet-pytorch/

Model trained by Gavin Ding
"""

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import wideresnet as wrn
from train import accuracy
from train import AverageMeter


# Data loading code
normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


def load_best_model(filename='cifar10_WideResNet.pth.tar'):
    """Load the best model"""
    # directory = "runs/{}/".format("WideResNet-28-10")
    # filename = directory + filename
    params = torch.load(filename)
    return params


def get_feature(model, invar):
    outvar = model.conv1(invar)
    outvar = model.block1(outvar)
    outvar = model.block2(outvar)
    outvar = model.block3(outvar)
    outvar = model.relu(model.bn1(outvar))
    outvar = F.avg_pool2d(outvar, 8)
    # outvar = outvar.view(-1, model.nChannels)
    # outvar = model.fc(outvar)
    return outvar


if __name__ == '__main__':

    model = wrn.WideResNet(depth=28, num_classes=10, widen_factor=4)
    param_dict = load_best_model()
    model.load_state_dict(param_dict["state_dict"])
    model = model.cuda()
    model.eval()


    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=100, shuffle=True, **{'num_workers': 1, 'pin_memory': True})

    top1 = AverageMeter()

    for i, (input, target) in enumerate(test_loader):
        target = target.cuda()
        output = model(Variable(input.cuda()))
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1[0], input.size(0))
        # break

    ftr = get_feature(model, Variable(input.cuda()))
