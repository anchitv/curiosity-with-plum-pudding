import torch
import math

def plumGauss(z):

    x0 = torch.squeeze(z)
    x1 = x0.transpose(0,1)

    batch = x0.size()[0]
    dim = x0.size()[1]  # vector length

    xx  = torch.bmm(x0.view(batch, 1, dim),
                    x0.view(batch, dim, 1)).squeeze(2)
    xx0 = xx.expand(batch,batch)
    xx1 = xx0.transpose(0,1)

    xy = xx0 + xx1 - 2*torch.matmul(x0,x1)

    xx2 = 2*xx*(torch.log(1+4*xx/dim)+1.3-math.log(5))/(1+1.6*xx/dim)

    result = 0.55*torch.sum(xx)/(1+2*dim) + 0.45*torch.sum(xx2)/(1+2*dim) \
           - 0.5*torch.sum(torch.log(1+xy*2/(1+2*dim)))/(batch-1)
    return math.sqrt(1+2*dim)*result/batch

def plumGauss0(z):

    x0 = torch.squeeze(z)
    x1 = x0.transpose(0,1)

    batch = x0.size()[0]
    dim = x0.size()[1]  # vector length

    xx  = torch.bmm(x0.view(batch, 1, dim),
                    x0.view(batch, dim, 1)).squeeze(2)
    xx0 = xx.expand(batch,batch)
    xx1 = xx0.transpose(0,1)

    xy = xx0 + xx1 - 2*torch.matmul(x0,x1)

    result = torch.sum(xx)/(1+2*dim) - 0.5*torch.sum(torch.log(1+xy*2/(1+2*dim)))/(batch-1)
    return math.sqrt(1+2*dim)*result/batch

def test():
    # test to see that it works
    z = torch.FloatTensor([[1,1,-1,1],[1,-1,1,1],[-1,-1,1,1]])
    loss = plumGauss(z)
    print(loss)
