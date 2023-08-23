import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as pl

from interferometer import Interferometer
from interferometer import Network
import lal

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import optax
import corner                            
#jax.config.update('jax_platform_name', 'cpu')

import distrax
import haiku as hk      #neural network library for JAX 
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
Array = jnp.ndarray
PRNGKey = Array
OptState = Any

from curses import KEY_REPLACE

from jimgw.PE.detector_preset import * 
from detector_projection import make_detector_response 

from jax.config import config
config.update("jax_enable_x64", True)


#GW routines 

def simulate_fd_sine_gaussian_waveform(A, t0, f0, tau, times, fmin, df):   #without phase for now
    
    t = times
    
    hpt = A*jnp.exp(-(t-t0)**2/tau**2)*jnp.cos(2*jnp.pi*f0*t) #time domain plus polarisation
    hct = A*jnp.exp(-(t-t0)**2/tau**2)*jnp.sin(2*jnp.pi*f0*t) #time domain cross polarisation

    hp = jnp.fft.rfft(hpt) #frequency domain plus polarisation
    hc = jnp.fft.rfft(hct) #frequency domain cross polarisation

    return hp, hc

def project_to_detector(detector, hp, hc, ra, dec, gmst_rad, psi):                   
    """Compute the response of the detector to incoming strain """
    return response[detector](jnp.atleast_2d(H1.freqs), hp, hc, ra, dec, gmst_rad, psi)


# likelihood class

class LogL(object):
    
    def __init__(self, true_gw_params):
        
        self.true_gw_params = true_gw_params
        self.detectors = {'H1': H1, 'L1': L1}
        
        gps_time = true_gw_params['t0']     #time of coalescence
        gps = lal.LIGOTimeGPS(gps_time)
        self.gmst_rad = lal.GreenwichMeanSiderealTime(gps)

        self.A = true_gw_params['A']
        self.t0 = true_gw_params['t0']
        self.f0 = true_gw_params['f0']
        self.tau = true_gw_params['tau']
        self.ra=jnp.atleast_1d(true_gw_params['ra'])
        self.dec = jnp.atleast_1d(true_gw_params['dec'])
        self.psi= jnp.atleast_1d(true_gw_params['psi'])

        self.f_min = 0.
        self.times = H1.times
        self.df = H1.df
        self.times2d = jnp.atleast_2d(H1.times)
        
        self.hp, self.hc = simulate_fd_sine_gaussian_waveform(self.A, self.t0, self.f0, self.tau, H1.times, self.f_min, H1.df)
        self.data = self.simulate_response(self.hp, self.hc, self.ra, self.dec, self.psi)
    
    def simulate_response(self, hp, hc, ra, dec, psi):
        r = {d: project_to_detector(d, hp, hc, ra, dec, self.gmst_rad, psi) for d in self.detectors.keys()}
        return r
    
    def __call__(self, params):
        hp, hc = simulate_fd_sine_gaussian_waveform(jnp.atleast_2d(params['A']).T, self.t0, self.f0, self.tau, self.times2d, self.f_min, self.df)
       
        r = self.simulate_response(hp, hc, params['ra'], params['dec'], params['psi'])

        residuals = jnp.array([r[ifo] - self.data[ifo] for ifo in self.detectors.keys()])
        
        return -jnp.real(jnp.sum(residuals*jnp.conj(residuals),axis=(0,2)))/2

    @property
    def params(self):
        params = ['A','ra','dec','psi']
        return params

    #@jax.jit
    def array_to_phys(self, x: Array) -> dict:
        
        p = dict()
        p['A'] = x[:,0]
        p['ra'] = x[:,1]*2*jnp.pi  #[0,2pi]
        p['dec'] = (x[:,2]-0.5)*jnp.pi  #[-pi/2,pi/2]
        p['psi'] = (x[:,3]-0.5)*jnp.pi #[-pi/2,pi/2]

        return p
    

#Training setup

from vi_routines import make_flow_model

@hk.without_apply_rng
@hk.transform
def sample_and_log_prob(prng_key: PRNGKey, n: int) -> Tuple[Any, Array]:

    model = make_flow_model(                          #this is the flow distribution (a distrax object)
        event_shape=(n_params,),
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins
    )

    return model.sample_and_log_prob(seed=prng_key, sample_shape=(n,))
     # returns x (sample from the flow q), and model.log_prob(x) (array of log(q) of th sampled points)

def log_prob(x: Array) -> Array:
    p = log_l.array_to_phys(x)
    return log_l(p) + jnp.log(jnp.cos(p['dec']))

def loss_fn(params: hk.Params, prng_key: PRNGKey, n: int) -> Array:       #computes reverse KL-divergence for the sample x_flow between the flow and gw loglikelihood.

    x_flow, log_q = sample_and_log_prob.apply(params, prng_key, n)           #gets sample from the flow and computes log_q for the sampled points.
    #log_p = log_likelihood(x_flow)
    log_p = log_prob(x_flow)
                                           #gets gw loglikelihood for the sampled points (after transforming them into physical space..)
    loss = jnp.mean(log_q - log_p)
    return loss

@jax.jit
def update(
    params: hk.Params,
    prng_key: PRNGKey,
    opt_state: OptState,
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, prng_key, Nsamps) #gradient w.r.t. params, evalueated at params, prng_key, Nsamps.
    updates, new_opt_state = optimiser.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


if __name__ == '__main__':

   
    #Problem setup
    f_min = 0. #minimum frequency cut-off
    T = 1.  #data segment duration (seconds)
    Fs = 100. #samplingfrequency  (Hz)
    t_start=0.

    #initialise the detectors
    H1 = Interferometer('H1','O3',f_min,T,Fs,t_start,20011997)
    L1 = Interferometer('L1','O3',f_min,T,Fs,t_start,27071994)

    true_params = dict(
            A=0.5,
            t0=0.5,
            f0=20.,
            tau=1/10,
            ra=jnp.atleast_1d(0.5),
            dec=jnp.atleast_1d(0.5),
            psi=jnp.atleast_1d(0.5))

    H1_response = make_detector_response(H1.tensor, H1.location)
    L1_response = make_detector_response(L1.tensor, L1.location)  
    response = {'H1': H1_response, 'L1': L1_response}

    # Compute signal SNR
    gps = lal.LIGOTimeGPS(true_params['t0'])
    gmst_rad = lal.GreenwichMeanSiderealTime(gps)

    L1.signal = L1_response(L1.freqs, hp, hc, true_params['ra'], true_params['dec'], gmst_rad, true_params['psi'])   
    H1.signal = H1_response(H1.freqs, hp, hc, true_params['ra'], true_params['dec'], gmst_rad, true_params['psi'])

    H1.psd=1.
    L1.psd=1.
    network = Network([H1,L1])
    print('Network SNR of the injected signal: ', network.snr())


    #flow parameters
    n_params = 4
    flow_num_layers = 4
    hidden_size = 16
    mlp_num_layers = 2
    num_bins = 6

    #training parameters
    epochs = 9000
    Nsamps = 1000

    learning_rate = 0.01
    optimiser = optax.adam(learning_rate)             #stochastic gradient descent 

    prng_seq = hk.PRNGSequence(42)
    key = next(prng_seq)
    params = sample_and_log_prob.init(key, prng_key=key, n=Nsamps)
    opt_state = optimiser.init(params)

    from tqdm import tqdm, trange
   
    ldict = dict(loss = [])
    training_stats = dict(loss = [], mean_loss = np.inf)

    log_l = LogL(true_params)      

    # Test likelihood
    l = log_l(true_params)
    print(f'log likelihood of true params = {l}')
    
    truths=[0.5,0.5,0.5,0.5]

    with trange(epochs) as tepochs:
        for epoch in tepochs:
            #update NN params (stochastic gradient descent with Adam optimiser)
            prng_key = next(prng_seq)
            loss = loss_fn(params,  prng_key, Nsamps)
            ldict['loss'] = f'{loss:.2f}'
            training_stats['loss'].append(loss)
        
            tepochs.set_postfix(ldict, refresh=True)
            params, opt_state = update(params, prng_key, opt_state)        #take a step in direction of stepest descent (negative gradient)

            #print results every 100 iterations (first one is plotted after 1st update.)
            if (epoch)%500 == 0:
                print(f'Epoch {epoch}, loss {loss}')
                x_gen, log_prob_gen = sample_and_log_prob.apply(params, next(prng_seq), 10*Nsamps)
                log_posterior = log_prob(x_gen)
                x_gen = np.array(x_gen, copy=False)
                p_gen = np.vstack(list(log_l.array_to_phys(x_gen).values()))
                data = np.concatenate([p_gen, [log_posterior]])
                fig = corner.corner(data.T, labels=['A','ra','dec','psi', 'log_p'], truths = [0.5,0.5,0.5,0.5,0.])
                pl.show()
                pl.savefig(f'flow_${epoch}.png')

            #if (epoch+1)%500 == 0:
                #stopping criterion
                #new_mean_loss = np.mean(training_stats['loss'][-500:])
                #if np.abs(training_stats['mean_loss']-(new_mean_loss))/training_stats['mean_loss'] < 0.001:
                #    break   #break here if the training loss has decreased by less then 1% in the last 500 steps. 
                #else:
                #    training_stats['mean_loss'] = new_mean_loss
    print("Done!")
    

    x_gen, log_prob_gen = sample_and_log_prob.apply(params, next(prng_seq), 100*Nsamps)
    x_gen = np.array(x_gen, copy=False)
    p_gen = np.vstack(list(log_l.array_to_phys(x_gen).values()))
    fig = corner.corner(p_gen.T, truths = truths)
    pl.show()
    pl.savefig(f'posterior_${epochs}.png')

    f = open('samples_emcee_cosprior.npy', 'rb')
    samples = np.load(f)


    kwargs = dict(
    bins=32,
    smooth=0.9,
    quantiles=[0.16, 0.5, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=True,
    plot_datapoints=True,
    fill_contours=True,
    show_titles=False,
    hist_kwargs=dict(density=True),
    range = [(0.2,1.2), (-0.5,1.6),(-1.2,1.4), (0,1.8), ])

    fig = None

    # Need to update colours for 1D and 2D plots
    kwargs["color"] = "C0"
    kwargs["hist_kwargs"]["color"] = "C0"
    fig = corner.corner(samples, fig=fig, **kwargs)

    kwargs["color"] = "C1"
    kwargs["hist_kwargs"]["color"] = "C1"
    fig = corner.corner(p_gen.T, labels=log_l.params, truths = truths, fig=fig, **kwargs)

    pl.show()
    pl.savefig(f'posterior_comparison_${epochs}.png')