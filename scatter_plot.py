import numpy as np
import matplotlib.pyplot as plt

##
fp1 = 'modereateDemo_eig.txt' 
fp2 = 'nonDemo_eig.txt' 

mod_data = np.loadtxt(fp1)
non_data = np.loadtxt(fp2)

figname = 'scatter_inertia.pdf'
plt.figure(figsize=(6,4))
plt.scatter(mod_data[:,0],mod_data[:,1],color='red',label = 'moderate')
plt.scatter(non_data[:,0],non_data[:,1],color='black',label = 'non Dementia')
plt.xlabel(r'$\lambda 1$')
plt.ylabel(r'$\lambda 2$')
plt.legend()
plt.tight_layout()
plt.savefig(figname)
