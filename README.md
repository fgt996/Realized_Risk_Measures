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
