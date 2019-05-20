import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser as AP


p = AP()
p.add_argument('--npy', type=str, help='Enter numpy file name to load')
p = p.parse_args()

coeff = np.load(p.npy)
coeff = torch.tensor(coeff).cuda()

lst = np.arange(0.1,1,0.1)

for i in range(len(lst)):
	sig=[]
	for j in range(coeff.shape[0]):
		a=torch.zeros(coeff[j].shape[0]).long().cuda()
	   	b=torch.arange(0, coeff[j].shape[0]).cuda()
		c=torch.where(((coeff[j] > -lst[i]) & (coeff[j] < lst[i])),a,b)
	        sig.append(torch.sum(c != 0).cpu().numpy())
	sig = np.array(sig)
	plt.plot(sig,label= lst[i])
	plt.legend()
plt.xlabel('iterations')
plt.ylabel('Significant zero components')
plt.show()
		
