# This is new mnist
# In this after taking a vector from top eigen subspace and transform it
# we now represent it using entire eigen subspace and see it if is only in top subspace
# pylint: disable = C, R, E1101, E1123
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from hessian import hessian
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser as AP


p = AP()
p.add_argument('--n_iter', type=int, default=100, help='Number of iterations of gradient descent')
p.add_argument('--device', type=int, default='0', help='PyTorch device string <device_name>:<device_id>')
p.add_argument('--lr', type=float, default=0.001, help='Learning rate for gradient descent')
p.add_argument('--fontsize', type=int, default=35, help='Font size for graph labels, ticks and legend.')
p.add_argument('--suffix', type=str, default='default', help='Add suffix to graph name')
p.add_argument('--range', type=float, default = 0.1, help='range to compare zero significant values')
p.add_argument('--top', type = int, default = 570, help = 'top eigen values to consider')
p = p.parse_args()


device = torch.device(p.device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(5, 5, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(5, 10)



    def forward(self, x):  # pylint: disable = W0221
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), x.size(1), -1).mean(-1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def train(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def make_epoch(epoch):
        model.train()
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(loader.dataset),
                    100. * i / len(loader), loss.item()))
    sig = []
    coeff = []
    for epoch in range(1, p.n_iter):
        make_epoch(epoch)
        if epoch % 10 == 0:
            h = compute_hessian(model, dataset)
            # h = (hessian, dominant eigen vec space, random vec in eigen subspace, entire eigen values)
            # analyse_hessian (model, top eigen space, top eigen vec)
            # sum, c = analyse_hessian(model, h[3], h[2])
            sum, c = analyse_hessian(model, h[0], h[2])
            sig.append(sum)
            if np.size(coeff) == 0:
                coeff = c.detach().cpu().numpy()
                coeff = np.expand_dims(coeff, axis=0)
            else:
                coeff = np.concatenate((coeff,np.expand_dims(c.detach().cpu().numpy(),axis=0)),0)
    sig = np.array(sig)
    plt.figure(figsize=(10, 8))
    plt.xlabel("Iterations", fontsize=p.fontsize)
    plt.ylabel("Number of Significant dimensions", fontsize=p.fontsize)
    plt.plot(sig)
    plt.savefig('sig_{}.png'.format(p.suffix), dpi=100)
    np.save(p.suffix,coeff)
    
# We will pass model, top-eigen space (inv_h), top eigen vector (top_vec)
def analyse_hessian(model, inv_h, top_vec):
    # Calculating the gradient in flatten form
    g=torch.rand(0).cuda(p.device).float()
    for i in model.parameters():
        for k, l in enumerate(i):
            if l.dim()==0:
                l=l.view(1)
                g=torch.cat((g.float(),torch.flatten(l)))
            else:
                g=torch.cat((g.float(),torch.flatten(l)))
    # Checking if invariant
    w_t = g- p.lr*top_vec
    # Since inv_h is a 570x20 dim we need to append 0's in all last columns
    if inv_h.shape[0] - inv_h.shape[1] != 0:
        inv_h = torch.cat((inv_h, torch.cuda.FloatTensor(inv_h.shape[0], inv_h.shape[0]-inv_h.shape[1]).fill_(0)),1)
    #coeff 
    coeff =  torch.mv(inv_h.transpose(0,1), w_t)
    # printing indices which satisfy a condition of staying within a range
    a=torch.zeros (coeff.shape[0]).long().cuda()
    b=torch.arange(0, coeff.shape[0]).cuda()
    c=torch.where(((coeff > - p.range) & (coeff < p.range)),a,b)
    print("Zero directions ", torch.sum(c == 0))
    return torch.sum(c == 0),coeff

def test(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=2000)
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1)  # get the index of the max log-probability
            correct += pred.eq(target).long().sum().item()

    print("accuracy = {}".format(correct*100.0 / len(loader.dataset)))

def compute_hessian(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=500)
    n = sum(p.numel() for p in model.parameters())
    h = torch.zeros(n, n, device=device,requires_grad=False)

    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum') / len(dataset)

        print'\rCompute the hessian: [{}]'.format("*" * i + " " * (len(loader) - i))
        hessian(loss, model.parameters(), out=h)
    print('\rCompute the hessian: [{}]'.format("*" * len(loader)))
    eigenvalues, eigenvec = torch.symeig(h,eigenvectors=True)
    '''
    print("The first eigenvalues are {}".format(eigenvalues[:20]))
    print("The last eigenvalues are {}".format(eigenvalues[-20:]))
    '''
    top = p.top
    dom = eigenvec[:,-top:]
    alpha=torch.rand(top, device=torch.device(p.device))
    vec=(alpha*dom).sum(1)
    vec=vec/torch.sqrt((vec*vec).sum())
    return (h, dom, vec, eigenvec)

def main():
    torch.backends.cudnn.benchmark = True
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST('../data', train=False, transform=transform)
    # make the example faster
    trainset = torch.utils.data.Subset(trainset, range(4000))
    model = Net().to(device)
    train(model, trainset)
    test(model, testset)
main()

