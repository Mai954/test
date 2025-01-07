import numpy as np
import matplotlib.pyplot as plt
import emcee
from matplotlib import pyplot
from astropy.modeling.models import Sersic1D
from scipy import integrate
###################################
# np.random.seed(123)
# # Choose the "true" parameters.
# m_true = -0.9594
# b_true = 4.294
# f_true = 0.534

# # Generate some synthetic data from the model.
# N = 50
# x = np.sort(10 * np.random.rand(N))
# # print(x)
# yerr = 0.1 + 0.5 * np.random.rand(N)
# y = m_true * x + b_true
# y += np.abs(f_true * y) * np.random.randn(N)
# y += yerr * np.random.randn(N)

G = 0.004301
obs_dis=np.array([300,245,180,289,211,191,176,202,191])
obs_dis_err=np.array([3,21,15,6,22,20,18,18,20])
r_obs=np.array([0.02225,0.05933,0.089,0.03125,0.07812,0.10938,0.14062,0.17188,0.20312])*6.535 ###in kpc scales

plt.errorbar(r_obs/6.535, obs_dis, yerr=obs_dis_err, fmt=".k", capsize=0)
x0 = np.linspace(0, 0.5, 500)
plt.plot(x0, np.sqrt(G*10**9*5/x0/6.535/1000), "-k", alpha=0.3, lw=3)# plot the curve with 5 times 10^9 Msun
plt.xlim(0, 0.3)
plt.ylim(50, 500)
plt.xlabel("arcsec")
plt.ylabel("Sigma");

# log prior
def lnprior(theta):
    Ie,re,n,mass_BH = theta
    if 1.0e10 <Ie< 1.0e12 and 0.1<re<3 and 0.4<n<10 and 1.0e8<mass_BH<1.0e10:
        return 0.0
    return -np.inf

# log likelihood function
def lnlike(theta, obs_dis,obs_dis_err,r_obs):
    Ie,re,n,mass_BH = theta
    
    s1 = Sersic1D(amplitude=Ie, r_eff=re)
    s1.n = n
    f = lambda x: s1(x)*x
    mass_bulge=np.zeros((len(r_obs)))
    for j in range (0,len(r_obs)):
        mass_bulge[j]=integrate.quad(f,0,r_obs[j])[0]  
    dis_bulge=G*mass_bulge/r_obs*2/3./1000
    dis_BH=G*mass_BH/r_obs*2/3./1000
    model = np.sqrt(dis_bulge+dis_BH)
    return -0.5 * np.sum((obs_dis - model)**2/obs_dis_err**2)

# log probability function
def lnprob(theta, obs_dis,obs_dis_err,r_obs):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,obs_dis,obs_dis_err,r_obs)

#Set up some walkers in a gaussian ball around the maximum likelyhood result
initial_guess=np.zeros((4))
initial_guess[0]=1.0e10# Ie, Msun
initial_guess[1]=1#re, kpc
initial_guess[2]=1# n
initial_guess[3]=5.0e9# mass_bh,Msun
# print(initial_guess)
num_params, num_walkers = 4, 100
# initial positions of the walkers
pos = [initial_guess+1.e-4*np.random.randn(num_params) for i in range(num_walkers)]
# print(pos)

# set up the sampler
sampler = emcee.EnsembleSampler(num_walkers, num_params, lnprob, args=(obs_dis,obs_dis_err,r_obs))
#and send it on a walk
# sampler.reset()
num_steps = 5000
pos0, prob, state = sampler.run_mcmc(pos, num_steps,progress=True)

#And plot the paths of the walkers
fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["Ie", "re", "n", "Mbh"]
for i in range(num_params):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

#Generate the Posterior distribution
import corner
samples = sampler.chain[:,:,:].reshape((-1, num_params))
print(samples.shape)

fig = corner.corner(samples, labels=labels)

#print the results of fitted parameters
from IPython.display import display, Math

for i in range(num_params):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

