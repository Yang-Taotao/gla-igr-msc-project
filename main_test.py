# %%
# Library import
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

# %%
# Test entry
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

# %%
# FIM density calc - 50x40
mcs = jnp.arange(1.0, 6.0, 0.1)
etas = jnp.arange(0.055, 0.255, 0.005)

gridpoints = it.product(mcs, etas)

density = []
for x in tqdm(gridpoints, total=len(mcs)*len(etas)):
    mceta = jnp.array([x[0], x[1], 0.0, 0.0])
    try:
        G = projected_fim(mceta)
        detg = jnp.sqrt(jnp.linalg.det(G))
    except AssertionError:
        detg = jnp.nan
    density.append(detg)

output = jnp.array(density).reshape([len(mcs), len(etas)])

# %%
# Log_10 based density data
log_density = []
for detg in density:
    log_detg = jnp.log10(detg)
    log_density.append(log_detg)

log_output = jnp.array(log_density).reshape([len(mcs), len(etas)])

# %% 
# Plot
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(
    mcs, 
    etas, 
    log_output.T,
    alpha=0.8,
    levels=10,
    cmap='gist_heat',
)
ax.set(
    xlabel=r'Chirp mass ($M_\odot$)', 
    ylabel=r'Symmetric mass ratio $\eta$',
)
cb = plt.colorbar(cs, ax=ax)
cb.ax.set_ylabel(r'$\log_{10}$ Template bank density')
plt.savefig("./figures/fig_06_fim_mc_mr_contour_log10.png")

# %%
