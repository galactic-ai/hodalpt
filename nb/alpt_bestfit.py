import numpy as np
import emcee
from hodalpt.sims import alpt as CS
from hodalpt.sims import quijote as Q
from hodalpt import stats
outdir = '/Users/mcc3842/CosmicSim2025/data/quijote/fiducial/1/alpt/'
theta_hod = {
    'logMmin': 13.06,
    'sigma_logM': 0.34,
    'logM0': 13.74,
    'logM1': 14.17,
    'alpha': 0.69,
    'Abias': 0.1, 
    'eta_conc': 1.12,
    'eta_cen': 0.3,
    'eta_sat': 1.26}
hod_quijote = Q.HODgalaxies(theta_hod,'/Users/mcc3842/CosmicSim2025/data/quijote/fiducial/1', z=0.5)
hod_Q_spec = stats.Pk_periodic(np.array(hod_quijote['Position']).T, Lbox=1000, Ngrid=256, Nmubin=20, fft='pyfftw', silent=True,rsd=2)

kk = hod_Q_spec['k'][:48]
Pref = hod_Q_spec['p0k'][:48]
frac_err = 0.05  # 5% errors everywhere why not 
sigma_Pref = frac_err * Pref

# deal with setup for log prior on theta alpt
theta0 = np.array([2.8845, 1.82277, -0.7889, 13.21866, 0.67392, 2.4e-05,0.7289, 1.39824, 1.3136, 0.4944])
theta_mu = theta0.copy()
theta_sigma = np.array([
    0.9,      # alpha
    0.55,     # beta
    0.4,      # dth
    5.0,      # rhoeps
    0.2,      # eps
    3.0e-06,  # nmean
    0.2,      # bv
    0.3,      # bb
    0.3,      # betarsd
    0.15      # gamma
])
param_names = np.array([
    "alpha","beta","dth","rhoeps","eps",
    "nmean","bv","bb","betarsd","gamma"
])
def P_alpt(theta):
    alpha, beta, dth, rhoeps, eps, nmean, bv, bb, betarsd, gamma = theta
    theta_alpt = dict(alpha=alpha, beta=beta, dth=dth,
                rhoeps=rhoeps, eps=eps, nmean=nmean)
    theta_bias = dict(bv=bv, bb=bb, betarsd=betarsd, gamma=gamma)
    xyz_g_del = CS.CSbox_galaxy(theta_alpt, theta_bias, outdir, silent=False)
    spec= stats.Pk_periodic(xyz_g_del.T, Lbox=1000, Ngrid=256, Nmubin=20, fft='pyfftw', silent=False,rsd=2)
    return spec['p0k'][:48]

def log_likelihood(theta,Pref, sigma_Pref): # chi2 likelihood bc don't know covmat
    P_model = P_alpt(theta)
    residuals = Pref - P_model
    chi2 = np.sum((residuals / sigma_Pref) ** 2)
    return -0.5 * chi2
def log_prior(theta):
    alpha, beta, dth, rhoeps, eps, nmean, bv, bb, betarsd, gamma = theta
    # hard bounds
    if not (0.0 < alpha < 10.0):       return -np.inf
    if not (0.0 < beta < 10.0):        return -np.inf
    if not (-5.0 < dth < 5.0):         return -np.inf
    if not (0.0 < rhoeps < 50.0):      return -np.inf
    if not (0.0 < eps < 5.0):          return -np.inf
    if not (1e-7 < nmean < 1e-3):      return -np.inf
    if not (0.0 < bv < 5.0):           return -np.inf
    if not (0.0 < bb < 5.0):           return -np.inf
    if not (0.0 < betarsd < 5.0):      return -np.inf
    if not (0.0 < gamma < 5.0):        return -np.inf
    diff = theta - theta_mu
    z = diff / theta_sigma

    return -0.5 * np.sum(z**2) - np.sum(np.log(np.sqrt(2*np.pi) * theta_sigma))
def log_posterior(theta,Pref, sigma_Pref):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, Pref,sigma_Pref)

# set up some chains

ndim = len(theta0)
nwalkers = 4 * ndim  

pos0 = theta0 + 1e-2 * theta0 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_posterior,
    args=(Pref, sigma_Pref)
)

# Run the chain
nsteps = 1000 
print("Running MCMC...")
chain = sampler.run_mcmc(pos0, nsteps, progress=True)[0]

chain = sampler.get_chain()                # shape: (nsteps, nwalkers, ndim)
log_prob = sampler.get_log_prob()         # shape: (nsteps, nwalkers)
accept_frac = sampler.acceptance_fraction # shape: (nwalkers,)
np.savez(
    "mcmc_alpt_chain.npz",
    chain=chain,
    log_prob=log_prob,
    accept_frac=accept_frac,
    theta0=theta0,
    theta_mu=theta_mu,
    theta_sigma=theta_sigma,
    param_names=param_names,
    k_data=kk,
    Pref_data=Pref,
    sigma_Pref=sigma_Pref,
)
print("Saved to mcmc_alpt_chain.npz")