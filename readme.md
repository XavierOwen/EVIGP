# Energetic Variational Gaussian Process Regression for Computer Experiments

author: Yuanxing Cheng, Lulu Kang, Yiwei Wang, Chun Liu

This is the code repository for the project: Energetic Variational Gaussian Process Regression for Computer Experiments. The code is written by Yuanxing Cheng and Yiwei Wang.

You can run `pip install -r requirements.txt` and `rscript r-requirement.r` first to check if the external libraries are ready.

## Quick Start

The easiest way to run the example is to type `python RMSPE_Toy.py` in the console, which uses saved RMSPE results to make a bar plot. If some data is missing, it will run the corresponding `.py` or `.r` to generate again using default parameters.

For now, there are 3 available examples. And can be tested using the following command

- `python RMSPE_Toy.py`
- `python RMSPE_Borehole.py`
- `python RMSPE_OTLcircuit.py`

## File structure

Folder `examples` include several .py files, and each can take several optional arguments. To run a Python one, for example, the toy example, with noninformative prior, constant mean function, and using mode to predict, type `python -m examples.Toy.EVIGP_toy_noninformative_constMean_mode` in the console. To see what arguments it can take, type `python -m examples.Toy.EVIGP_toy_noninformative_constMean_mode -h`. To run an R one, for example, we can type in console `rscript ./examples/Toy/GPfit_toy_constMean.r`.

Each Python code will create a `.npy` file for the values of RMSPE in the folder `RMSPE` and a figure relevant in the folder `figs`. Each R code will create a `.csv` file for RMSPE in the same folder.
