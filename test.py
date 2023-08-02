import jax.numpy as jnp
from data import gw_ripple, gw_fisher, gw_plotter
from data.gw_config import f_sig, f_psd, mc_repo, mr_repo, theta_repo
from data.gw_fisher import fim
from data.gw_ripple import gradient_plus_mceta, waveform_plus_mceta, waveform_plus_normed

from matplotlib import pyplot as plt

mc = 20
eta = 0.2
mceta = jnp.array([mc, eta])

#wf = waveform_plus_normed(mceta)
#print(wf)
g = gradient_plus_mceta(mceta)
print(g)
plt.plot(g[:,0].real)
plt.plot(g[:,0].imag)
plt.show()
G = fim(mceta)

density = []
mcs = jnp.arange(5,20,0.1)
for mc in mcs:
    mceta = jnp.array([mc, eta])
    G = fim(mceta)
    detg = jnp.sqrt(jnp.linalg.det(G))
    density.append(detg)

plt.plot(mcs, density)
