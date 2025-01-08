import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import emcee
from matplotlib import pyplot
from astropy.modeling.models import Sersic1D
from scipy import integrate
from multiprocessing import Pool
from multiprocessing import cpu_count
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

# import os
# os.environ["OMP_NUM_THREADS"] = "1"
############load the observed data###################
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
############define the likelihood function##############
def lnprior(theta):
    Ie,re,n,mass_BH = theta
    if 1.0e10 <Ie< 1.0e12 and 0.1<re<3 and 0.4<n<10 and 1.0e8<mass_BH<1.0e10:
        return 0.0
    return -np.inf

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

def lnprob(theta, obs_dis,obs_dis_err,r_obs):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,obs_dis,obs_dis_err,r_obs)
############run the emcee###############################
#Set up walkers
initial_guess=np.zeros((4))
initial_guess[0]=1.0e10# Ie, Msun
initial_guess[1]=1#re, kpc
initial_guess[2]=1# n
initial_guess[3]=5.0e9# mass_bh,Msun
num_params, num_walkers = 4, 50
pos = [initial_guess+1.e-4*np.random.randn(num_params) for i in range(num_walkers)]

with Pool() as pool:
    sampler = emcee.EnsembleSampler(num_walkers, num_params, lnprob, args=(obs_dis,obs_dis_err,r_obs),pool=pool)
    #burn-in
    start = time.time()
    state = sampler.run_mcmc(pos,500,progress=True)
    sampler.reset()
    sampler.reset()
    num_steps = 500
    sampler.run_mcmc(state, num_steps,progress=True)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    # print("{0:.1f} times faster than serial".format(serial_time / multi_time))

############plot the results###############################
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
# fig.savefig('home/mailiao/dispersion_profile/walkers.pdf', bbox_inches='tight')

#Generate the Posterior distribution
import corner
samples = sampler.chain[:,:,:].reshape((-1, num_params))
print(samples.shape)
fig = corner.corner(samples, labels=labels)
# fig.savefig('home/mailiao/dispersion_profile/distributions.pdf', bbox_inches='tight')

#print and save the results for each fitted parameters
from IPython.display import display, Math

results=np.zeros((4,3))
for i in range(num_params):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    results[i,:]=mcmc[1], q[0], q[1]
    display(Math(txt))
