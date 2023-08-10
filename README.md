## gla-igr-msc-project
This is the degree project for *MSc in Astrophysics* at *University of Glasgow*
- Initialized: May 30, 2023
- Editted: August 10, 2023

## Environment
```WSL: Ubuntu```

## Dependecies
- ```jax```
- ```ripplegw```
- ```bilby```

## Test GW params
```python
m1, m2 = 36.0, 29.0
s1, s2 = 0.0, 0.0
dist_mpc = 40.0
c_time, c_phas = 0.0, 0.0
ang_inc, ang_pol = 0.0, 0.0
```

## Purpose
- Using ```jax``` ```ripplegw``` ```bilby``` to construct machine learning scripts
- Use ```jax``` with ```CUDA``` support for ```@jax.jit``` compilations
- Use normalizing flow for template bank placement analysis

## File structure
```bash
.
├── LICENSE
├── README.md
├── data
│   ├── __init__.py
│   ├── gw_cfg.py
│   ├── gw_fim.py
│   ├── gw_plt.py
│   └── gw_rpl.py
├── figures
├── legacy
│   ├── data_legacy
│   │   └── gw_plotter.py
│   ├── figures_legacy
│   │   ├── fig_01_ripple_waveform.png
│   │   ├── fig_02_ripple_waveform_grad.png
│   │   ├── fig_03_bilby_psd.png
│   │   ├── fig_04_fim_heatmap.png
│   │   ├── fig_05_fim_mc_mr.png
│   │   ├── fig_06_fim_mc_mr_contour.png
│   │   ├── fig_06_fim_mc_mr_contour_alternative.png
│   │   └── fig_06_fim_mc_mr_contour_log10.png
│   └── main_legacy.py
└── main.py
```

## Legacy plots
- GW150914 waveform generated with ```ripplegw```
<p align="center">
  <img src="./legacy/figures/fig_01_ripple_waveform.png"/>
</p>

- GW150914 waveform gradient plot with ```jax.vmap(jax.grad())```
<p align="center">
  <img src="./legacy/figures/fig_02_ripple_waveform_grad.png"/>
</p>

- PSD aLIGO noise curve with ```bilby```
<p align="center">
  <img src="./legacy/figures/fig_03_bilby_psd.png"/>
</p>

- Fisher Information Matrix for test GW params
<p align="center">
  <img src="./legacy/figures/fig_04_fim_heatmap.png"/>
</p>

- Fisher Information Matrix wrt mc and mr
<p align="center">
  <img src="./legacy/figures/fig_05_fim_mc_mr.png"/>
</p>

- Fisher Information Matrix contour plot
<p align="center">
  <img src="./legacy/figures/fig_06_fim_mc_mr_contour.png"/>
</p>

## Active plots
- Fisher Information Matrix contour plot at log_10 base
<p align="center">
  <img src="./legacy/figures/fig_06_fim_mc_mr_contour_log10.png"/>
</p>
