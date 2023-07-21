## gla-igr-msc-project
This is a degree project for *MSc in Astrophysics* at *University of Glasgow*
- Initialized: May 30, 2023
- Editted: July 21, 2023

## Environment
- WSL: Ubuntu

## Test GW params
```python
m1, m2 = 36.0, 29.0
s1, s2 = 0.0, 0.0
dist_mpc = 40.0
c_time, c_phas = 0.0, 0.0
ang_inc, ang_pol = 0.0, 0.0
```

## Purpose
- Using JAX and other packages to construct machine learning scripts
- Use JAX with CUDA support for faster compilations
- Use normalizing flow for template bank placement problems

## File structure
```bash
.
├── README.md
├── data
│   ├── __init__.py
│   ├── gw_fisher.py
│   ├── gw_plotter.py
│   └── gw_ripple.py
├── data_cache
├── figures
├── main.py
└── notebook
    └── main_legacy.py
```

## Active plots
- GW170817 waveform generated with ```ripplegw```
![GW170817 waveform](./figures/fig_01_ripple_waveform.png)
- GW170817 waveform gradient plot with ```jax.vmap(jax.grad())```
![Gradient plot](./figures/fig_02_ripple_waveform_grad.png)
- PSD aLIGO noise curve with ```bilby```
![PSD plot](./figures/fig_03_bilby_psd.png)
- Fisher Information Matrix for test GW params 
![FIM plot](./figures/fig_04_fim.png)
