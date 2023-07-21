## gla-igr-msc-project
This is a degree project for *MSc in Astrophysics* at *University of Glasgow*
- Initialized: May 30, 2023
- Editted: July 11, 2023

## Environment
- WSL: Ubuntu

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
│   ├── data_36.0_29.0_0.0_0.0_40.0_0.0_0.0_0.0_0.0_cros.npy
│   ├── data_36.0_29.0_0.0_0.0_40.0_0.0_0.0_0.0_0.0_plus.npy
│   ├── grad_36.0_29.0_0.0_0.0_40.0_0.0_0.0_0.0_0.0_cros.npy
│   └── grad_36.0_29.0_0.0_0.0_40.0_0.0_0.0_0.0_0.0_plus.npy
├── figures
│   ├── fig_01_ripple_waveform.png
│   ├── fig_02_ripple_waveform_grad.png
│   ├── fig_03_bilby_psd.png
│   └── fig_04_fim.png
├── main.py
└── notebook
    ├── main.ipynb
    └── main_legacy.py
```

## Active plots
- GW170817 waveform generated with 
```ripplegw```
![GW170817 waveform](./figures/fig_01_ripple_waveform.png)
- GW170817 waveform gradient plot with
```jax.vmap(jax.grad())```
![Gradient plot](./figures/fig_02_ripple_waveform_grad.png)
```bilby```
![PSD plot](./figures/fig_03_bilby_psd.png)