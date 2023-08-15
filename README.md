# gla-igr-msc-project

This is the degree project for *MSc in Astrophysics* at *University of Glasgow*

- Initialized: May 30, 2023
- Editted: August 15, 2023

## Environment

```WSL: Ubuntu```

## Dependecies

- ```jax```
- ```ripplegw```
- ```bilby```

## Normalizing flow dependencies

- ```haiku```
- ```distrax```
- ```optax```

## Test GW params

```python
m1, m2 = 36.0, 29.0
s1, s2 = 0.0, 0.0
dist_mpc = 40.0
c_time, c_phas = 0.0, 0.0
ang_inc, ang_pol = 0.0, 0.0
```

## Purpose

- Using ```jax``` ```ripplegw``` ```bilby``` to construct FIM calculation script
- Use ```jax``` with ```CUDA``` support for ```@jax.jit``` compilations
- Employ ```jax.grad()``` for automatic gradient calculations
- Use normalizing flow for approximating template bank placement density

## File structure

```bash
.
├── LICENSE
├── README.md
├── data
│   ├── __init__.py
│   ├── __jaxcache__
│   ├── __pycache__
│   ├── gw_cfg.py
│   ├── gw_fim.py
│   ├── gw_plt.py
│   ├── gw_rpl.py
│   ├── vi_jax.py
│   └── vi_routines.py
├── figures
│   └── fig_01_fim_contour_mc_eta_log10.png
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
![Test GW Waveform](./legacy/figures_legacy/fig_01_ripple_waveform.png)

- GW150914 waveform gradient plot with ```jax.vmap(jax.grad())```
![Test GW Waveform Gradient](./legacy/figures_legacy/fig_02_ripple_waveform_grad.png)

- PSD aLIGO noise curve with ```bilby```
![Detector PSD](./legacy/figures_legacy/fig_03_bilby_psd.png)

- Fisher Information Matrix for test GW params
![Test FIM Heat Map](./legacy/figures_legacy/fig_04_fim_heatmap.png)

- Fisher Information Matrix wrt chirp mass and symmetric mass ratio
![FIM 1-D](./legacy/figures_legacy/fig_05_fim_mc_mr.png)

- Fisher Information Matrix contour plot
![Density Contour Plot](./legacy/figures_legacy/fig_06_fim_mc_mr_contour.png)

## Active plots

- Fisher Information Matrix contour plot at log_10 base
![Projected Density Contour PLot](./legacy/figures_legacy/fig_06_fim_mc_mr_contour_log10.png)
