#%% Importing the necessary libraries and preparing the environment

import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Our modules
import utils
import models

import warnings
warnings.filterwarnings('ignore')

random_seed = 2

N_sims = 50_000 #Number of paths to simulate in the Monte Carlo approach
dh_H = 0.5 #Hurst exponent for the DH approach
Thetas = [0.05, 0.025, 0.01] #Confidence levels for the VaR and ES
EWMA_windows = [5, 21, 252]
output_path = 'synthetic_output/' #Path to save the results

#------------- Load a real dataset to use its timestemps
data_path = 'YOUR_DATA_PATH/EXAMPLE_STOCK.txt'
df = pd.read_csv(data_path, header=None)
df['datetime'] = pd.to_datetime(df[0] + ' ' + df[1], format='%m/%d/%Y %H:%M')
df = df.set_index('datetime')
df.index = pd.to_datetime(df.index)
df.columns = ['Day', 'Minute', 'Open', 'High', 'Low', 'Close', 'Volume']
df = utils.Fill_RTH_Minutes(df)
df['day'] = [str(val.month)+'_'+str(val.day) for val in df.index]
#------------- End loading
#------------- Alternative loading - If you don't have a real dataset, just use:
'''
intraday = pd.timedelta_range("09:00:00", "16:30:00", freq="1min") #Regualr Trading Hours
biz_days = pd.date_range(
    datetime(2009,1,1), datetime(2019,12,31), freq="B") # Business days
idx = biz_days.repeat(len(intraday)) +\
    pd.Index(intraday.tolist() * len(biz_days)) #Trading minutes in business days

# Example empty DataFrame with that index
df = pd.DataFrame(index=idx)
df['day'] = [str(val.month)+'_'+str(val.day) for val in df.index]
'''
#------------- End alternative loading

#%% Define functions for simulating data and computing the risk

class ProcessGenerator():
    def __init__(self, dist):
        self.dist = dist
        if self.dist == 'norm':
            from scipy.stats import norm
            self.rvs = norm.rvs
        elif self.dist == 't':
            from scipy.stats import t as stud_t
            self.rvs = stud_t.rvs
    
    def sim_iid(self, N, pars):
        if self.dist == 'norm':
            mu, sigma = pars
            return self.rvs(loc=mu, scale=sigma, size=N)
        elif self.dist == 't':
            nu, mu, sigma = pars
            return self.rvs(nu, loc=mu, scale=sigma, size=N)
    
    def sim_ma1(self, N, pars):
        if self.dist == 'norm':
            phi, mu, sigma = pars
            X = self.rvs(loc=mu, scale=sigma, size=N+1)
        elif self.dist == 't':
            phi, nu, mu, sigma = pars
            X = self.rvs(nu, loc=mu, scale=sigma, size=N+1)
        return X[1:] + phi*X[:-1]

def gaussian_tail_stats(theta, loc=0., scale=1.):
    '''
    Compute the Value at Risk and Expected Shortfall for a Gaussian distribution
    INPUTS:
        theta (float): the quantile to compute the statistics
        loc (float): the mean of the distribution
        scale (float): the standard deviation of the distribution

    OUTPUTS:
        dict: a dictionary with the following keys_ 'var' (float; the Value at Risk), 'es' (float; the Expected Shortfall)
    '''
    from scipy.stats import norm

    return {
        'var':loc + scale*norm.ppf(theta),
        'es':loc - scale*norm.pdf(norm.ppf(1-theta))/theta }

def simulate_iid(N, c, nu, mu, sigma, ant_v=True, seed=None):
    '''
    Monte-Carlo simulation for computing daily return with iid
    t-distributed intra-day returns.

    INPUTS:
        - N: int
            the number of Monte-Carlo paths to simulate (if ant_v==True, the number is doubled).
        - c: int
            the number of time indexes to sample.
        - nu: float
            the degrees of freedom.
        - mu: float
            the location parameter of the t-distribution.
        - sigma: float
            the scale parameter of the t-distribution.
        - ant_v: bool, optional
            Flag to indicate if the antithetic variates should be used. Default is True.
        - seed: int, optional
            Seed for the random number generator. Default is None.

    OUTPUTS:
        - samples: ndarray
            the simulated low-frequency returns.
    '''
    from scipy.stats import t as stud_t #Load the Student's t-distribution

    np.random.seed(seed)
    if ant_v:
        out = stud_t.rvs(nu, loc=0, scale=sigma, size=N*c).reshape(N,c)
        samples = (mu + np.concatenate([out, -out], axis=0)).sum(axis=1)
    else:
        out = stud_t.rvs(nu, loc=mu, scale=sigma, size=N*c).reshape(N,c)
        samples = out.sum(axis=1)
    return samples
    
def simulate_ma(N, c, phi, nu, mu, sigma, ant_v=True, seed=None):
    '''
    Monte-Carlo simulation for computing daily return with MA(1)
    intra-day returns and t-distributed innovations.

    INPUTS:
        - N: int
            the number of Monte-Carlo paths to simulate (if ant_v==True, the number is doubled).
        - c: int
            the number of time indexes to sample.
        - phi: float
            the autoregressive coefficient.
        - nu: float
            the degrees of freedom.
        - mu: float
            the location parameter of the t-distribution.
        - sigma: float
            the scale parameter of the t-distribution.
        - ant_v: bool, optional
            Flag to indicate if the antithetic variates should be used. Default is True.
        - seed: int, optional
            Seed for the random number generator. Default is None.

    OUTPUTS:
        - samples: ndarray
            the simulated low-frequency returns.
    '''
    from scipy.stats import t as stud_t #Load the Student's t-distribution

    np.random.seed(seed)
    if ant_v:
        inn = stud_t.rvs(nu, loc=0, scale=sigma, size=N*(c+1)).reshape(N,c+1)
        inn = np.concatenate([inn, -inn], axis=0) + mu
    else:
        inn = stud_t.rvs(nu, loc=mu, scale=sigma, size=N*(c+1)).reshape(N,c+1)
    return phi*inn[:,0] + inn[:,-1] + (1+phi)*(inn[:,1:-1].sum(axis=1))

#%% Core loop - Filter VaR and ES

Gen_processes = [
    ProcessGenerator('norm').sim_iid, ProcessGenerator('norm').sim_ma1,
    ProcessGenerator('t').sim_iid, ProcessGenerator('t').sim_ma1]

'''
Distribution parameters previously fitted on real data.
We report the median across the fitted values day-by-day.
'''

'''
Parameters fitted (likelihood) assuming data iid Gaussian
c: [mean, std]
'''
iid_N_values = {
    39:[0.0, 0.002088418118073098],
    78:[0.0, 0.0015151753371827734],
    130:[0.0, 0.0011979036187315776]
}
'''
Parameters fitted (likelihood) assuming data iid t-distributed
c: [degrees of freedom, mean, std]
'''
iid_t_values = {
    39:[2.0693339743377113, -4.46885174527257e-05, 0.001352010314769584],
    78:[2.0276062352731214, -2.675302367306937e-05, 0.0009522128610519072],
    130:[2.00440027812995, -1.6982296390329673e-05, 0.0007348562721830217]
}
'''
Parameters fitted (likelihood) assuming data MA(1) with Gaussian innovation
c: [MA coefficient, mean, std]
'''
ma_N_values = {
    39:[-0.06080969182007847, 2.951771741873321e-07, 0.002040810156555649],
    78:[-0.04903058584070752, -9.868673990942999e-09, 0.0014933894319432734],
    130:[-0.051004856454001615, -1.1065287221872441e-10, 0.0011850857589384759]
}
'''
Parameters fitted (likelihood) assuming data MA(1) with t-distributed innovation
c: [MA coefficient, degrees of freedom, mean, std]
'''
ma_t_values = {
    39:[-0.05016513562986328, 2.115614078844289, -2.7044834565108583e-05, 0.001354434659998953],
    78:[-0.05004337647353343, 2.0457064531109914, -1.7850045250910122e-05, 0.0009679481555702513],
    130:[-0.05332505460429271, 2.011118081457462, -1.1786663234013637e-05, 0.0007411877264963962]
}

Gen_params = {
    'iid_N':iid_N_values, 'ma_N':ma_N_values,
    'iid_t':iid_t_values, 'ma_t':ma_t_values}

for idx2use in range(len(Gen_params)):
    gen_process = Gen_processes[idx2use]
    gen_name = list(Gen_params.keys())[idx2use]
    print('Working with', gen_name)

    # Load the results if they have already been computed
    try:
        with open(f'{output_path}idx_{idx2use}.pickle', 'rb') as f:
            out_res = pickle.load(f)
    except:
        out_res = dict()

    # Define a "-1" year used only for computing the historical mean
    year  = 2009
    if year not in out_res.keys(): #Check if the dictionray has already been initialized
        out_res[year] = dict()

    for c in [39, 78, 130]:
        if c not in out_res[year].keys(): #Check if the dictionray has already been initialized
            out_res[year][c] = dict()
            out_res[year][c]['states'] = dict()

            temp_df = df[(df.index>=datetime(year, 1, 1)) &\
                        (df.index<datetime(year+1, 1, 1)) ]
            tot_days = len(temp_df.day.unique())
            t_sub = utils.Subordinator(c, sub_type='clock')
            
            np.random.seed(year)
            y_price = pd.Series(
                gen_process(tot_days*(c+1), Gen_params[gen_name][c]),
                index=temp_df.index[np.concatenate([
                    n_d*391+np.array(t_sub.predict(temp_df.iloc[:391])) for n_d in range(tot_days)])])
            
            y_price.index = pd.to_datetime(y_price.index)
            y_price = np.exp(y_price.cumsum())
            out_res[year][c]['y_price'] = y_price
            out_res[year][c]['y'] =\
                utils.IntraSeries_2_DayReturn(y_price)

    for year in range(2010, 2020):
        if year not in out_res.keys(): #Check if the dictionray has already been initialized
            out_res[year] = dict()

        for c in [39, 78, 130]:
            if c not in out_res[year].keys(): #Check if the dictionray has already been initialized
                out_res[year][c] = dict()
                out_res[year][c]['states'] = dict()

                temp_df = df[(df.index>=datetime(year, 1, 1)) &\
                            (df.index<datetime(year+1, 1, 1)) ]
                tot_days = len(temp_df.day.unique())
                t_sub = utils.Subordinator(c, sub_type='clock')
                
                np.random.seed(year)
                y_price = pd.Series(
                    gen_process(tot_days*(c+1), Gen_params[gen_name][c]),
                    index=temp_df.index[np.concatenate([
                        n_d*391+np.array(t_sub.predict(temp_df.iloc[:391])) for n_d in range(tot_days)])])
                
                y_price.index = pd.to_datetime(y_price.index)
                y_price = np.exp(y_price.cumsum())
                out_res[year][c]['y_price'] = y_price
                out_res[year][c]['y'] =\
                    utils.IntraSeries_2_DayReturn(y_price)
                
            hist_mu = out_res[year-1][c]['y'].mean()

            for theta in Thetas:
                if theta not in out_res[year][c].keys(): #Check if the dictionray has already been initialized
                    out_res[year][c][theta] = dict()

            sub = 'iid'
            # log-returns assumed to be iid; 0 prior on the drift
            if sub not in out_res[year][c]['states'].keys(): #Check if the computation has already been done; otherwise, do it
                out_res[year][c]['states'][sub] = utils.price2params(
                    out_res[year][c]['y_price'],
                    c=c, mu_prior=0, sub_type='identity', ma=False)
            
            # log-returns assumed to be iid; EWMA prior on the drift
            for win_len in EWMA_windows:
                if sub+str(win_len) not in out_res[year][c]['states'].keys(): #Check if the computation has already been done; otherwise, do it
                    out_res[year][c]['states'][sub+str(win_len)] = utils.price2params(
                        out_res[year][c]['y_price'], c=c, mu_prior=2/(win_len+1),
                        sub_type='identity', ma=False, hist_mean=hist_mu/c)

            sub = 'ma'
            # log-returns assumed follow an MA(1) process; 0 prior on the drift
            if sub not in out_res[year][c]['states'].keys(): #Check if the computation has already been done; otherwise, do it
                out_res[year][c]['states'][sub] = utils.price2params(
                    out_res[year][c]['y_price'],
                    c=c, mu_prior=0, sub_type='identity', ma=True)
                
            # log-returns assumed follow an MA(1) process; EWMA prior on the drift
            for win_len in EWMA_windows:
                if sub+str(win_len) not in out_res[year][c]['states'].keys(): #Check if the computation has already been done; otherwise, do it
                    out_res[year][c]['states'][sub+str(win_len)] = utils.price2params(
                        out_res[year][c]['y_price'], c=c, mu_prior=2/(win_len+1),
                        sub_type='identity', ma=True, hist_mean=hist_mu/c)
                    
            # Save the progress
            with open(f'{output_path}idx_{idx2use}.pickle', 'wb') as f:
                pickle.dump(out_res, f)
        print('Finished Fitting t-distribution')

        #------------------------------ DH Approach ------------------------------
        sub = 'dh'
        for c in [39, 78, 130]:
            for theta in Thetas:
                if sub not in out_res[year][c][theta].keys(): #Check if the computation has already been done; otherwise, do it
                    q_daily_pred, es_daily_pred =\
                        models.DH_RealizedRisk(
                            c, theta, dh_H, sub_type='identity').fit(
                            out_res[year][c]['y_price'])

                    out_res[year][c][theta][sub] =\
                        {'qr':q_daily_pred, 'er':es_daily_pred}
        print('Finished DH')
        # Save the progress
        with open(f'{output_path}idx_{idx2use}.pickle', 'wb') as f:
            pickle.dump(out_res, f)

        #------------------------------ CH Approach iid ------------------------------
        sub = 'iid_ch'
        for c in [39, 78, 130]:
            for theta in Thetas:
                if sub not in out_res[year][c][theta].keys(): #Check if the computation has already been done; otherwise, do it
                    q_daily_pred, es_daily_pred =\
                        models.Ch_RealizedRisk(theta, ma=False).fit(
                            c, out_res[year][c]['states'][sub[:-3]], jobs=2)

                    out_res[year][c][theta][sub] =\
                        {'qr':q_daily_pred, 'er':es_daily_pred}
                    
            for win_len in EWMA_windows:
                for theta in Thetas:
                    if sub+str(win_len) not in out_res[year][c][theta].keys(): #Check if the computation has already been done; otherwise, do it
                        q_daily_pred, es_daily_pred =\
                            models.Ch_RealizedRisk(theta, ma=False).fit(
                                c, out_res[year][c]['states'][sub[:-3]+str(win_len)], jobs=2)

                        out_res[year][c][theta][sub+str(win_len)] =\
                            {'qr':q_daily_pred, 'er':es_daily_pred}
        print('Finished CH iid')
        # Save the progress
        with open(f'{output_path}idx_{idx2use}.pickle', 'wb') as f:
            pickle.dump(out_res, f)

        #------------------------------ CH Approach MA ------------------------------
        sub = 'ma_ch'
        for c in [39, 78, 130]:
            for theta in Thetas:
                if sub not in out_res[year][c][theta].keys(): #Check if the computation has already been done; otherwise, do it
                    q_daily_pred, es_daily_pred =\
                        models.Ch_RealizedRisk(theta, ma=True).fit(
                            c, out_res[year][c]['states'][sub[:-3]], jobs=2)

                    out_res[year][c][theta][sub] =\
                        {'qr':q_daily_pred, 'er':es_daily_pred}
                    
            for win_len in EWMA_windows:
                for theta in Thetas:
                    if sub+str(win_len) not in out_res[year][c][theta].keys(): #Check if the computation has already been done; otherwise, do it
                        q_daily_pred, es_daily_pred =\
                            models.Ch_RealizedRisk(theta, ma=True).fit(
                                c, out_res[year][c]['states'][sub[:-3]+str(win_len)], jobs=2)

                        out_res[year][c][theta][sub+str(win_len)] =\
                            {'qr':q_daily_pred, 'er':es_daily_pred}
        print('Finished CH MA')
        # Save the progress
        with open(f'{output_path}idx_{idx2use}.pickle', 'wb') as f:
            pickle.dump(out_res, f)

        #-------------------------- MonteCarlo Approach iid --------------------------
        sub = 'iid_mc'
        for c in [39, 78, 130]:
            if sub not in out_res[year][c][Thetas[-1]].keys(): #Check if the computation has already been done; otherwise, do it
                q_daily_pred, es_daily_pred = models.MC_RealizedRisk(
                    Thetas, ma=False).fit(
                        N_sims, c,
                        out_res[year][c]['states'][sub[:-3]],
                        ant_v=True, seed=random_seed)

                for theta in Thetas:
                    out_res[year][c][theta][sub] =\
                        {'qr':q_daily_pred[theta], 'er':es_daily_pred[theta]}
                    
            for win_len in EWMA_windows:
                if sub+str(win_len) not in out_res[year][c][Thetas[-1]].keys(): #Check if the computation has already been done; otherwise, do it
                    q_daily_pred, es_daily_pred = models.MC_RealizedRisk(
                        Thetas, ma=False).fit(
                            N_sims, c,
                            out_res[year][c]['states'][sub[:-3]+str(win_len)],
                            ant_v=True, seed=random_seed)

                    for theta in Thetas:
                        out_res[year][c][theta][sub+str(win_len)] =\
                            {'qr':q_daily_pred[theta], 'er':es_daily_pred[theta]}
            # Save the progress
            with open(f'{output_path}idx_{idx2use}.pickle', 'wb') as f:
                pickle.dump(out_res, f)

        print('Finished MonteCarlo iid')

        #-------------------------- MonteCarlo Approach ma --------------------------
        sub = 'ma_mc'
        for c in [39, 78, 130]:
            if sub not in out_res[year][c][Thetas[-1]].keys(): #Check if the computation has already been done; otherwise, do it
                q_daily_pred, es_daily_pred = models.MC_RealizedRisk(
                    Thetas, ma=True).fit(
                        N_sims, c,
                        out_res[year][c]['states'][sub[:-3]],
                        ant_v=True, seed=random_seed)

                for theta in Thetas:
                    out_res[year][c][theta][sub] =\
                        {'qr':q_daily_pred[theta], 'er':es_daily_pred[theta]}
                    
            for win_len in EWMA_windows:
                if sub+str(win_len) not in out_res[year][c][Thetas[-1]].keys(): #Check if the computation has already been done; otherwise, do it
                    q_daily_pred, es_daily_pred = models.MC_RealizedRisk(
                        Thetas, ma=True).fit(
                            N_sims, c,
                            out_res[year][c]['states'][sub[:-3]+str(win_len)],
                            ant_v=True, seed=random_seed)

                    for theta in Thetas:
                        out_res[year][c][theta][sub+str(win_len)] =\
                            {'qr':q_daily_pred[theta], 'er':es_daily_pred[theta]}
            # Save the progress
            with open(f'{output_path}idx_{idx2use}.pickle', 'wb') as f:
                pickle.dump(out_res, f)

        print('Finished MonteCarlo ma')

        #-------------------------- Ensemble Ch-MC --------------------------
        for sub in ['iid', 'ma']:
            for c in [39, 78, 130]:
                if sub+'_en' not in out_res[year][c][Thetas[-1]].keys(): #Check if the computation has already been done; otherwise, do it
                    for theta in Thetas:
                        temp_qr, temp_er = dict(), dict()
                        for key in out_res[year][c][theta][sub+'_ch']['qr'].keys():
                            if np.isnan(out_res[year][c][theta][sub+'_ch']['qr'][key]):
                                temp_qr[key] = out_res[year][c][theta][sub+'_mc']['qr'][key]
                            else:
                                temp_qr[key] = (
                                    out_res[year][c][theta][sub+'_ch']['qr'][key] +\
                                    out_res[year][c][theta][sub+'_mc']['qr'][key] ) / 2

                        for key in out_res[year][c][theta][sub+'_ch']['er'].keys():
                            if np.isnan(out_res[year][c][theta][sub+'_ch']['er'][key]):
                                temp_er[key] = out_res[year][c][theta][sub+'_mc']['er'][key]
                            else:
                                temp_er[key] = (
                                    out_res[year][c][theta][sub+'_ch']['er'][key] +\
                                    out_res[year][c][theta][sub+'_mc']['er'][key] ) / 2

                        out_res[year][c][theta][sub+'_en'] = {'qr':temp_qr, 'er':temp_er}

                for win_len in EWMA_windows:
                    if sub+'_en'+str(win_len) not in out_res[year][c][Thetas[-1]].keys(): #Check if the computation has already been done; otherwise, do it
                        for theta in Thetas:
                            temp_qr, temp_er = dict(), dict()
                            for key in out_res[year][c][theta][sub+'_ch'+str(win_len)]['qr'].keys():
                                if np.isnan(out_res[year][c][theta][sub+'_ch'+str(win_len)]['qr'][key]):
                                    temp_qr[key] = out_res[year][c][theta][sub+'_mc'+str(win_len)]['qr'][key]
                                else:
                                    temp_qr[key] = (
                                        out_res[year][c][theta][sub+'_ch'+str(win_len)]['qr'][key] +\
                                        out_res[year][c][theta][sub+'_mc'+str(win_len)]['qr'][key] ) / 2

                            for key in out_res[year][c][theta][sub+'_ch'+str(win_len)]['er'].keys():
                                if np.isnan(out_res[year][c][theta][sub+'_ch'+str(win_len)]['er'][key]):
                                    temp_er[key] = out_res[year][c][theta][sub+'_mc'+str(win_len)]['er'][key]
                                else:
                                    temp_er[key] = (
                                        out_res[year][c][theta][sub+'_ch'+str(win_len)]['er'][key] +\
                                        out_res[year][c][theta][sub+'_mc'+str(win_len)]['er'][key] ) / 2

                            out_res[year][c][theta][sub+'_en'+str(win_len)] = {'qr':temp_qr, 'er':temp_er}
        # Save the progress
        with open(f'{output_path}idx_{idx2use}.pickle', 'wb') as f:
            pickle.dump(out_res, f)

        print('Finished Ensemble Ch-MC')

        #------------------------ Ground Truth ------------------------
        sub = 'GT'
        for c in [39, 78, 130]:
            if sub not in out_res[year][c][Thetas[-1]].keys():
                q_daily_pred, es_daily_pred = dict(), dict()
                for theta in Thetas:
                    q_daily_pred[theta], es_daily_pred[theta] = dict(), dict()
                #--------- load curr_int_state only to have its keys, that are the days
                curr_int_state = out_res[year][c]['states']['iid']
                
                # Divide the two cases
                if gen_name[-1] == 'N':
                    for theta in Thetas:
                        if gen_name[:3] == 'iid':
                            N_risk = gaussian_tail_stats(
                                theta, c*Gen_params[gen_name][c][0],
                                np.sqrt(c)*Gen_params[gen_name][c][1])
                        else:
                            phi = Gen_params[gen_name][c][0]
                            N_risk = gaussian_tail_stats(
                                theta, c*(1+phi)*Gen_params[gen_name][c][1],
                                np.sqrt(
                                    (c-1)*(1+phi)*(1+phi) + 1 + phi*phi
                                )*Gen_params[gen_name][c][2])
                            
                        q_val_temp = N_risk['var']
                        e_val_temp = N_risk['es']
                        for temp_key in curr_int_state.keys():
                            q_daily_pred[theta][temp_key] = q_val_temp
                            es_daily_pred[theta][temp_key] = e_val_temp
                else:
                    if gen_name[:3] == 'iid':
                        # Eventually adjust the coefficients
                        np.random.seed(random_seed)
                        sim_data = simulate_iid(
                            N_sims, c, *Gen_params[gen_name][c], ant_v=True)
                    else:
                        # Eventually adjust the coefficients
                        np.random.seed(random_seed)
                        sim_data = simulate_ma(
                            N_sims, c, *Gen_params[gen_name][c], ant_v=True)
                        
                    for theta in Thetas:
                        q_val_temp = np.quantile(sim_data, theta)
                        e_val_temp = sim_data[ sim_data <= q_val_temp ].mean()
                        for temp_key in curr_int_state.keys():
                            q_daily_pred[theta][temp_key] = q_val_temp
                            es_daily_pred[theta][temp_key] = e_val_temp

                for theta in Thetas:
                    out_res[year][c][theta][sub] =\
                        {'qr':q_daily_pred[theta], 'er':es_daily_pred[theta]}
            # Save the progress
            with open(f'{output_path}idx_{idx2use}.pickle', 'wb') as f:
                pickle.dump(out_res, f)

        print('\n')
        print('Finished year', year, '\n\n\n')

#%% Compute the distance from the Ground Truth

def rMSE(x, y):
    return np.sqrt(np.nanmean(np.square(np.array(x)-np.array(y))))

Algos = ['dh', 'iid_en', 'ma_en', 'iid_ch', 'ma_ch', 'iid_mc', 'ma_mc']

Assets = [
    'iid_N', 'ma_N', 'iid_t', 'ma_t']
Assets_N = ['iid_N', 'ma_N']
Assets_t = ['iid_t', 'ma_t']

# Load the filtered risk measures
out_res = dict()
for n_a, asset in enumerate(Assets):
    with open(f'{output_path}idx_{n_a}.pickle', 'rb') as f:
        temp_res = pickle.load(f)
    out_res[asset] = temp_res

# Define the years
Years = dict()
for asset in Assets:
    Years[asset] = range(2010, 2020)

#-------------------- Step 1: VaR Table
Thetas = [0.05, 0.025, 0.01]

print('\\begin{tabular}{lccccccccc}')
print('\\toprule')
print("\\multicolumn{10}{c}{\\textbf{Gaussian Synthetic Dataset}} \\\\")
print('\\toprule')
print('\multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{3}{c}{$\\theta=0.05$} & \multicolumn{3}{c}{$\\theta=0.025$} & \multicolumn{3}{c}{$\\theta=0.01$} \\\\')
print('\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}')
print(' & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ \\\\')
print('\\midrule')

row2print = dict()

for sub in Algos:
    row2print[sub] = '\\_'.join(sub.split('_'))

for theta in Thetas:
    for c in [39, 78, 130]:
        top_q, top_loss_q = list(), np.inf
        sec_q, sec_loss_q = list(), np.inf
        print_q = dict()
        for sub in Algos:
            print_q[sub] = 0
            tot_runs = 0
            for asset in Assets_N:
                tot_runs += len(Years[asset])
                for year in Years[asset]:
                    print_q[sub] += rMSE(
                        list(out_res[asset][year][c][theta][sub]['qr'].values()),
                        list(out_res[asset][year][c][theta]['GT']['qr'].values())
                        )
            print_q[sub] /= tot_runs

            if print_q[sub] < top_loss_q:
                sec_q, sec_loss_q = top_q, top_loss_q
                top_q, top_loss_q = [sub], print_q[sub]
            elif print_q[sub] == top_loss_q:
                top_q.append(sub)
            elif print_q[sub] < sec_loss_q:
                sec_q, sec_loss_q = [sub], print_q[sub]
            elif print_q[sub] == sec_loss_q:
                sec_q.append(sub)
        
        if len(top_q) > 1: sec_q=list()
        for sub in Algos:
            if sub in top_q:
                row2print[sub] += ' & \\textbf{' + f'{round(1e3*print_q[sub], 3):.3f}' + '}'
            elif sub in sec_q:
                row2print[sub] += ' & \\underline{' + f'{round(1e3*print_q[sub], 3):.3f}' + '}'
            else:
                row2print[sub] += ' & ' + f'{round(1e3*print_q[sub], 3):.3f}'
        
for sub in Algos:
    print(row2print[sub] + '\\\\')
    if sub[:3] == 'vol':
        print('\\midrule')

print('\\bottomrule')
print('\\end{tabular}')
print('')
print('\\vspace{0.1cm}')
print('')

print('\\begin{tabular}{lccccccccc}')
print('\\toprule')
print("\\multicolumn{10}{c}{\\textbf{Student's t Synthetic Dataset}} \\\\")
print('\\toprule')
print('\multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{3}{c}{$\\theta=0.05$} & \multicolumn{3}{c}{$\\theta=0.025$} & \multicolumn{3}{c}{$\\theta=0.01$} \\\\')
print('\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}')
print(' & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ \\\\')
print('\\midrule')

row2print = dict()

for sub in Algos:
    row2print[sub] = '\\_'.join(sub.split('_'))

for theta in Thetas:
    for c in [39, 78, 130]:
        top_q, top_loss_q = list(), np.inf
        sec_q, sec_loss_q = list(), np.inf
        print_q = dict()
        for sub in Algos:
            print_q[sub] = 0
            tot_runs = 0
            for asset in Assets_t:
                tot_runs += len(Years[asset])
                for year in Years[asset]:
                    print_q[sub] += rMSE(
                        list(out_res[asset][year][c][theta][sub]['qr'].values()),
                        list(out_res[asset][year][c][theta]['GT']['qr'].values())
                        )
            print_q[sub] /= tot_runs

            if print_q[sub] < top_loss_q:
                sec_q, sec_loss_q = top_q, top_loss_q
                top_q, top_loss_q = [sub], print_q[sub]
            elif print_q[sub] == top_loss_q:
                top_q.append(sub)
            elif print_q[sub] < sec_loss_q:
                sec_q, sec_loss_q = [sub], print_q[sub]
            elif print_q[sub] == sec_loss_q:
                sec_q.append(sub)
        
        if len(top_q) > 1: sec_q=list()
        for sub in Algos:
            if sub in top_q:
                row2print[sub] += ' & \\textbf{' + f'{round(1e3*print_q[sub], 3):.3f}' + '}'
            elif sub in sec_q:
                row2print[sub] += ' & \\underline{' + f'{round(1e3*print_q[sub], 3):.3f}' + '}'
            else:
                row2print[sub] += ' & ' + f'{round(1e3*print_q[sub], 3):.3f}'
        
for sub in Algos:
    print(row2print[sub] + '\\\\')
    if sub[:3] == 'vol':
        print('\\midrule')

print('\\bottomrule')
print('\\end{tabular}')
print('')

#-------------------- Step 2: ES Table
Thetas = [0.05, 0.025]

print('\\begin{tabular}{lcccccc||cccccc}')
print('\\toprule')
print(" & \multicolumn{6}{c}{Gaussian Synthetic Dataset} & \multicolumn{6}{c}{Student's t Synthetic Dataset} \\\\")
print('\cmidrule(lr){2-7} \cmidrule(lr){8-13}')
print('\multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{3}{c}{$\\theta=0.05$} & \multicolumn{3}{c}{$\\theta=0.025$} & \multicolumn{3}{c}{$\\theta=0.05$} & \multicolumn{3}{c}{$\\theta=0.025$} \\\\')
print('\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}')
print(' & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ \\\\')
print('\\midrule')

row2print = dict()

for sub in Algos:
    row2print[sub] = '\\_'.join(sub.split('_'))

for theta in Thetas:
    for c in [39, 78, 130]:
        top_q, top_loss_q = list(), np.inf
        sec_q, sec_loss_q = list(), np.inf
        print_q = dict()
        for sub in Algos:
            print_q[sub] = 0
            tot_runs = 0
            for asset in Assets_N:
                tot_runs += len(Years[asset])
                for year in Years[asset]:
                    print_q[sub] += rMSE(
                        list(out_res[asset][year][c][theta][sub]['er'].values()),
                        list(out_res[asset][year][c][theta]['GT']['er'].values())
                        )
            print_q[sub] /= tot_runs

            if print_q[sub] < top_loss_q:
                sec_q, sec_loss_q = top_q, top_loss_q
                top_q, top_loss_q = [sub], print_q[sub]
            elif print_q[sub] == top_loss_q:
                top_q.append(sub)
            elif print_q[sub] < sec_loss_q:
                sec_q, sec_loss_q = [sub], print_q[sub]
            elif print_q[sub] == sec_loss_q:
                sec_q.append(sub)
        
        if len(top_q) > 1: sec_q=list()
        for sub in Algos:
            if sub in top_q:
                row2print[sub] += ' & \\textbf{' + f'{round(1e2*print_q[sub], 3):.3f}' + '}'
            elif sub in sec_q:
                row2print[sub] += ' & \\underline{' + f'{round(1e2*print_q[sub], 3):.3f}' + '}'
            else:
                row2print[sub] += ' & ' + f'{round(1e2*print_q[sub], 3):.3f}'

for theta in Thetas:
    for c in [39, 78, 130]:
        top_q, top_loss_q = list(), np.inf
        sec_q, sec_loss_q = list(), np.inf
        print_q = dict()
        for sub in Algos:
            print_q[sub] = 0
            tot_runs = 0
            for asset in Assets_t:
                tot_runs += len(Years[asset])
                for year in Years[asset]:
                    print_q[sub] += rMSE(
                        list(out_res[asset][year][c][theta][sub]['er'].values()),
                        list(out_res[asset][year][c][theta]['GT']['er'].values())
                        )
            print_q[sub] /= tot_runs

            if print_q[sub] < top_loss_q:
                sec_q, sec_loss_q = top_q, top_loss_q
                top_q, top_loss_q = [sub], print_q[sub]
            elif print_q[sub] == top_loss_q:
                top_q.append(sub)
            elif print_q[sub] < sec_loss_q:
                sec_q, sec_loss_q = [sub], print_q[sub]
            elif print_q[sub] == sec_loss_q:
                sec_q.append(sub)
        
        if len(top_q) > 1: sec_q=list()
        for sub in Algos:
            if sub in top_q:
                row2print[sub] += ' & \\textbf{' + f'{round(1e2*print_q[sub], 3):.3f}' + '}'
            elif sub in sec_q:
                row2print[sub] += ' & \\underline{' + f'{round(1e2*print_q[sub], 3):.3f}' + '}'
            else:
                row2print[sub] += ' & ' + f'{round(1e2*print_q[sub], 3):.3f}'
        
for sub in Algos:
    print(row2print[sub] + '\\\\')
    if sub[:3] == 'vol':
        print('\\midrule')

print('\\bottomrule')
print('\\end{tabular}')
print('')

