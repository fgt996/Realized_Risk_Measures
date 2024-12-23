# Realized Risk Measures


# Motivation

This repository provides a framework for filtering risk measures, namely Value at Risk (VaR) and Expected Shortfall (ES) from high-frequency data. The former is defined as the quantile of the conditional returns distribution, and the latter is the tail mean, i.e., the mean of all the values below the quantile. Mathematically, let's consider a probability level $`\theta\in[0,1]`$, assume to be at time 0, and consider a horizon $`T>0`$. In the following, we consider time 0 as the starting point of the regular trading hour of a given day, that is 09:30, and $T$ the end of the day, that is 16:00. Let $`(\Omega, \mathcal{F}, \mathbb{P})`$ be a probability space and $`\{\mathcal{F}_t\}_{t=0,T}`$ be a filtration. $`\mathcal{F}_0`$ represents the information set at the starting time of the considered period. Let $`Y`$ be the asset log return distribution, $`F_Y`$ its Cumulative Density Function (CDF), and $`F^{-1}_Y`$ the quantile function. Then, the VaR and ES at level $`\theta`$ are defined as: $`VaR_\theta(Y) = F_Y^{-1}(\theta)`$ and $`ES_\theta(Y) = \mathbb{E}[Y | Y \le VaR_\theta]`$.

Specifically, the target risk measures are on the daily timescale. The general pipeline can be summarized as follows. Firstly, we work on a single day at once, thus treating them as separate episodes. Moreover, we focus on the minute-by-minute data on the regular trading hours, thus neglecting the overnight dynamics where different dynamics enter the game. As the regular trading hour is from 9:30 to 16:00, there are 390 observations each day. Let $`\{S^{(t)}_i\}_{i=0}^{390}`$ be the intra-day price series of a specific day $`t`$. Let $`Y^{(t)}:=\log(S^{(t)}_{390})-\log(S^{(t)}_0)`$ be the daily return. Our aim is to filter VaR and ES of $`Y^{(t)}`$ conditional on the information set at time 0 (that is, at the beginning of the regular trading hour). The first step is to aggregate the intra-day data in a coarser granularity by using a subordinator based on the market intensity. The idea behind this step is that by aggregating the data in this way, we should be able to recover a more regular and easy-to-handle process. A subordinator can be thought of as a transformation of the clock time, that is, an injective map $`\tau:\{0, 1, \cdots, c\} \rightarrow \{0,1,\cdots,390\}`$, with $`\tau(0)=0`$ and $`\tau(c)=390`$. After having applied it, $`c+1\approx60`$ observations are obtained: $`\{S^{(t)}_{\tau(j)}\}_{j=0}^c`$ and the corresponding log returns are computed: $`\{Y^{(t)}_j\}_{j=1}^c`$, with $`Y^{(t)}_j = \log(S^{(t)}_{\tau(j)}) - \log(S^{(t)}_{\tau(j-1)})`$. Moreover, our code also allows for a further preprocessing step based on a Moving Average (MA) filter. Then, a fat tail distribution is fitted to the cleaned intra-day returns. The last step consists of scaling from intra-day to daily granularity. This task is straightforward under the Gaussianity assumption but could be tricky otherwise. However, we have identified two parallel ways to achieve this goal. From an analytical perspective, it is possible to employ the characteristic function to aggregate the returns in the Fourier space and then come back by using the Gil-Pelaez theorem. From a simulation point of view, the Monte Carlo approach can be used to simulate the daily distribution starting from the intra-day one. Both of them are valid and have pros and cons, so we opted for an equally-weighted ensemble.

# Getting Started

First, download and unzip the repository. Then, move to the download folder and install the conda environment:
```bash
cd Realized_Risk_Measures-main
conda env create -f RRM_env.yml
conda activate RRM
```

Then, move to the ```code``` directory and run the code. All the code is assumed to be run from the ```code``` folder.
```bash
cd code
```

# What is in this Repository

This repository shares the code used for the experimental stage of "A High-frequency Approach to Risk Measures" [1]. Specifically, the aim is to provide a range of tools to filter the daily risk measures starting from the minute-by-minute observations. It is worth underlining that we are working with the left tail of the return time series in a financial context.

The repository is organized as follows:

### utils
The ```utils``` module contains some functions used for the data preprocessing.

The most relevant class is ```Subordinator```, which is a way to aggregate the high-frequency data into coarser granularities. It exploits the market activity to clean the microstructural noise.

The ```Fill_RTH_Minutes``` function fills the holes and imputes the nan from a DataFrame made up of minute-by-minute prices in order to have significant values for the regular trading hours (from 09:30 to 16:00) of all the days in the original DataFrame.

```IntraSeries_2_DayReturn``` returns an np.array made up of the daily returns (logarithmic, computed as $`\log(S_{16:00}) - \log(S_{09:30})`$) for every day in the DataFrame.

```price2params``` maps a pd.DataFrame into a dictionary. Every key corresponds to a day in the original DataFrame, every value corresponds to the parameters of a Student's t-distribution fitted to the subordinated returns (so, the ```Subordinator``` is applied inside it and the subordinated returns are assumed to be iid).

Finally, ```price2params_ma``` is the same, but it assumes that the subordinated returns follow an MA(1) process, so the first element in every value is the autoregressive coefficient, and the last entries are the t parameters fitted on the innovations series.

### models

The ```models``` module is the core of this repository. It contains five classes for filtering the risk measures.

```MC_RealizedRisk``` is one of the filtering approaches proposed in [1]. It is based on the assumption of iid subordinated returns following the t-distribution. The Monte-Carlo approach is used to recover the low-frequency risk measures.

```MC_RealizedRisk_MA```is another approach proposed in [1]. It is the same as ```MC_RealizedRisk```, but the subordinated returns are assumed to follow an MA(1) process.

```Ch_RealizedRisk``` is similar to ```MC_RealizedRisk```, but it exploits an approach based on the high-frequency characteristic function to obtain the low-frequency risk measures.

```Ch_RealizedRisk_MA``` is the same of ```MC_RealizedRisk_MA```, but it is based on the characteristic function.

```DH_RealizedRisk``` is the approach proposed in [2]. It is based on the assumption of self-similarity of the subordinated logarithmic price process.

Please refer to [1] for a comprehensive description of these approaches and their pro and cons.

An example of filtering the risk measures from a pandas.Series is:
```python
import numpy as np
from utils import price2params
from models import MC_RealizedRisk

c = 78 #Number of intra-day returns to sample
theta = 0.05 #Desired probability level
N_sims = 50_000 #Number of paths for the Monte-Carlo simulation

y_price = pd.read_csv(file) #load the price time series

params_dict = price2params(y_price, c=c, sub_type='tpv') #Fit the t-distribution for every day in y_price.index

mdl = MC_RealizedRisk(theta) # Initialize the model
res = mdl.fit(N_sims, c, params_dict, ant_v=True, seed=2) #Compute the realized risk measures
print(res['qr']) #Print the realized VaR
print(res['er']) #Print the realized ES
```

### example_experiment
```example_experiment.py``` provides an example of a comparison between the different filtering approaches. It is the mould of the experiments carried out in [1]. Note that, according to copyright constraints, we cannot share the data used.

# Bibliography
[1]: Gatta, F., & Lillo, F., & Mazzarisi, P. (2024). A High-frequency Approach to Risk Measures. TBD.

[2]: Dimitriadis, T., & Halbleib, R. (2022). Realized quantiles. Journal of Business & Economic Statistics, 40(3), 1346-1361.

# Similar Repository
If you are interested in risk measures, you can find interesting the [![Static Badge](https://img.shields.io/badge/CAESar%20repository-blue?style=plastic)](https://github.com/fgt996/CAESar) It contains a collection of approaches for forecasting VaR and ES.

