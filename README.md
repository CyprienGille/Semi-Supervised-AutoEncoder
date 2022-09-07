# Semisupervised Autoencoder

This repository contains the code from :

> Artificial Intelligence for Semi-supervised classification in biomedical applications using a new supervised autoencoder, Gille C. and Guyard F. and Barlaud M., (2022). https://arxiv.org/abs/2208.10315 .

## Repository contents

 - `semisupervised_tests.py` : This is the main script used to produce the results shown in the paper. It generates plots in the `plots` directory, and saves results (metrics, losses...) as CSVs in the `results_semi` folder. All parameters are tunable near the start of the script.
 - `param_plots` : This is a helper script to reproduce the plots from Figures 2 and 3 of the aforementioned paper.
 - `eta_optimization.py` : This script is used to find the optimal sparsification parameter $\eta$ either by dichotomy or using the [golden section strategy](https://en.wikipedia.org/wiki/Golden-section_search).
 - `functions` : Contains function utilities useful for the other main scripts.
 - `data` : Contains the two datasets presented in the paper.
 - `plots` and `results_semi` are results directories filled by executing `semisupervised_tests.py`.
