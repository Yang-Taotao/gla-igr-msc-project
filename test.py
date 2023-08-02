import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from data import gw_ripple, gw_fisher, gw_plotter
from data.gw_config import f_sig, f_psd, mc_repo, mr_repo, theta_repo
from data.gw_fisher import fim, projected_fim
from data.gw_ripple import gradient_plus, waveform_plus_restricted, waveform_plus_normed
import jax.numpy as jnp

from matplotlib import pyplot as plt
import itertools as it
from tqdm import tqdm

mc = 20
eta = 0.2
mceta = jnp.array([mc, eta])
fisher_params = jnp.array([mc, eta, 0.0, 0.0])

wf = waveform_plus_restricted(fisher_params, f_sig)
print(wf.shape)
g = gradient_plus(fisher_params)
print(g.shape)
# print(g)
# plt.plot(g[:,0].real)
# plt.plot(g[:,0].imag)
# plt.show()
G = projected_fim(fisher_params)
print("Projected FIM: ",G)

mcs = jnp.arange(3,10,0.1)
etas = jnp.arange(0.05,0.25,0.005)

gridpoints = it.product(mcs,etas)

density = []
for x in tqdm(gridpoints, total=len(mcs)*len(etas)):
    mceta = jnp.array([x[0],x[1],0.0,0.0])
    try:
        G = projected_fim(mceta)
        detg = jnp.sqrt(jnp.linalg.det(G))
    except AssertionError:
        detg = jnp.nan
    density.append(detg)

output = jnp.array(density).reshape([len(mcs),len(etas)])
plt.contourf(mcs, etas, output.T)
plt.colorbar().ax.set_ylabel('Template bank density')
plt.xlabel('Chirp mass ($M_\odot$)')
plt.ylabel('Symmetric mass ratio $\eta$')
plt.savefig('projected_metric.png')
