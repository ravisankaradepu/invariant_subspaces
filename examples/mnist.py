# pylint: disable = C, R, E1101, E1123
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from hessian import hessian


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(loader.dataset),
                    100. * i / len(loader), loss.item()))

    for epoch in range(1, 20):
        make_epoch(epoch)


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

    print("accuracy = {}".format(correct / len(loader.dataset)))


def compute_hessian(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=500)

    n = sum(p.numel() for p in model.parameters())
    h = torch.zeros(n, n, device=device)

    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum') / len(dataset)

        print'\rCompute the hessian: [{}]'.format("*" * i + " " * (len(loader) - i))
        hessian(loss, model.parameters(), out=h)
    print(model.parameters())
    print('\rCompute the hessian: [{}]'.format("*" * len(loader)))
    print(h.shape)
    print('Diagonalize the hessian...    ')
    eigenvalues, eigenvec = torch.symeig(h)
    print("The first eigenvalues are {}".format(eigenvalues[:20]))
    print("The last eigenvalues are {}".format(eigenvalues[-20:]))
    top = 20
    dom = eigenvec[:,-top:]
    alpha=torch.rand(top, device=torch.device("cuda"))
    vec=(alpha*dom).sum(1)
    vec=vec/torch.sqrt((vec*vec).sum())
    # inverse of hessian
    inv_h=torch.pinverse(h)
    return (h, inv_h, vec)


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
    # top_vec is the vector obtained by drawing a vector from top eigen subspace of Hessian
    #Hess, top_vec = compute_hessian(model, trainset)
    h = compute_hessian(model, trainset)
    inv_h = h[1]
    top_vec = h[2]
    # Calculating the gradient in flatten form
    g=torch.rand(0).cuda().float()
    for i in model.parameters():
        for k, l in enumerate(i):
            if l.dim()==0:
                l=l.view(1)
                g=torch.cat((g.float(),torch.flatten(l)))
            else:
                g=torch.cat((g.float(),torch.flatten(l)))
    
    # Checking if invariant
    w_t = g-0.001*top_vec
    #coeff 
    coeff =  torch.mv(inv_h, w_t)
    print(coeff)
main()
