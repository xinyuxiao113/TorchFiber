import numpy as np, matplotlib.pyplot as plt


error = np.random.randn(100000, 2) # your data 


x_range=(-3, 3)
plt.figure(figsize=(8,6), dpi=200)
plt.hist(error[:,0], bins=100, range=x_range, density=True, alpha=0.5, label='real')
plt.hist(error[:,1], bins=100, range=x_range, density=True, alpha=0.5, label='imag')

# plot N(0,1) pdf
x = np.linspace(*x_range, 100)
y = np.exp(-x**2/2) / np.sqrt(2*np.pi)
plt.plot(x, y, label='N(0,1)')

plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('Error distribution of the testing data (error = Rx - Tx - PBC)')