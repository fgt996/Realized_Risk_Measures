#%% Importing the necessary libraries and preparing the environment

import pickle
import numpy as np
import pandas as pd
from datetime import datetime

import utils
import models

import warnings
warnings.filterwarnings('ignore')

random_seed = 2

N_sims = 50_000 #Number of paths to simulate in the Monte Carlo approach
dh_H = 0.5 #Hurst exponent for the DH approach
Thetas = [0.05, 0.025, 0.01] #Confidence levels for the VaR and ES
output_path = 'output/bank/' #Path to save the results

files_list = ['BAC.txt', 'MS.txt', 'C.txt', 'BK.txt',
              'STT.txt', 'JPM.txt', 'GS.txt', 'WFC.txt'] #Files to be processed

Years = dict() #Years to be processed
for file in files_list:
    if file not in ['MS.txt', 'GS.txt']:
        Years[file] = range(1998, 2021)
    elif file == 'MS.txt':
        Years[file] = range(2006, 2021)
    elif file == 'GS.txt':
        Years[file] = range(2000, 2021)

#%% Core loop - Filter VaR and ES

for file in files_list:
    print('Working with', file)
    # Load the results if they have already been computed
    try:
        with open(f'{output_path}{file.split(".")[0]}.pickle', 'rb') as f:
            out_res = pickle.load(f)
    except:
        out_res = dict()

    # Load the data and preprocess it
    df = pd.read_csv('data/bank/'+file, header=None)
    df['datetime'] = pd.to_datetime(df[0] + ' ' + df[1], format='%m/%d/%Y %H:%M')
    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    df.columns = ['Day', 'Minute', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = utils.Fill_RTH_Minutes(df)

    for year in Years[file]:
        y_train = df[(df.index>=datetime(year, 1, 1)) &\
                        (df.index<datetime(year+1, 1, 1))].Close
        y_vols = df[(df.index>=datetime(year, 1, 1)) &\
                        (df.index<datetime(year+1, 1, 1))].Volume
        hist_mu = df[(df.index>=datetime(year-1, 1, 1)) &\
                        (df.index<datetime(year, 1, 1))].Close
        hist_mu = utils.IntraSeries_2_DayReturn(hist_mu).mean()

        if year not in out_res.keys(): #Check if the dictionray has already been initialized
            out_res[year] = dict()
            y_daily = utils.IntraSeries_2_DayReturn(y_train)
            out_res[year]['y'] = y_daily

        for c in [39, 78, 130]:
            if c not in out_res[year].keys(): #Check if the dictionray has already been initialized
                out_res[year][c] = dict()
                out_res[year][c]['states'] = dict()
            for theta in Thetas:
                if theta not in out_res[year][c].keys(): #Check if the dictionray has already been initialized
                    out_res[year][c][theta] = dict()

            for sub in ['clock_iid', 'tpv_iid', 'vol_iid']:
                if sub not in out_res[year][c]['states'].keys(): #Check if the computation has already been done; otherwise, do it
                    out_res[year][c]['states'][sub] = utils.price2params(
                        y_train, c=c, mu=hist_mu/c, sub_type=sub[:-4], vol=y_vols)

            for sub in ['clock_ma', 'tpv_ma', 'vol_ma']:
                if sub not in out_res[year][c]['states'].keys(): #Check if the computation has already been done; otherwise, do it
                    out_res[year][c]['states'][sub] = utils.price2params_ma(
                        y_train, c=c, mu=hist_mu/c, sub_type=sub[:-3], vol=y_vols)
                    
            # Save the progress
            with open(f'{output_path}{file.split(".")[0]}.pickle', 'wb') as f:
                pickle.dump(out_res, f)
        print('Finished Fitting t-distribution')

        #------------------------------ DH Approach ------------------------------
        for sub in ['clock_dh', 'tpv_dh', 'vol_dh']:
            for c in [39, 78, 130]:
                for theta in Thetas:
                    if sub not in out_res[year][c][theta].keys(): #Check if the computation has already been done; otherwise, do it
                        q_daily_pred, es_daily_pred =\
                            models.DH_RealizedRisk(
                                c, theta, dh_H, sub_type=sub[:-3]).fit(
                                y_train, vol=y_vols)

                        out_res[year][c][theta][sub] =\
                            {'qr':q_daily_pred, 'er':es_daily_pred}
        print('Finished DH')
        # Save the progress
        with open(f'{output_path}{file.split(".")[0]}.pickle', 'wb') as f:
            pickle.dump(out_res, f)

        #------------------------------ CH Approach iid ------------------------------
        for sub in ['clock_iid_ch', 'tpv_iid_ch', 'vol_iid_ch']:
            for c in [39, 78, 130]:
                for theta in Thetas:
                    if sub not in out_res[year][c][theta].keys(): #Check if the computation has already been done; otherwise, do it
                        q_daily_pred, es_daily_pred =\
                            models.Ch_RealizedRisk(theta).fit(
                                c, out_res[year][c]['states'][sub[:-3]], jobs=2)

                        out_res[year][c][theta][sub] =\
                            {'qr':q_daily_pred, 'er':es_daily_pred}
        print('Finished CH iid')
        # Save the progress
        with open(f'{output_path}{file.split(".")[0]}.pickle', 'wb') as f:
            pickle.dump(out_res, f)

        #------------------------------ CH Approach MA ------------------------------
        for sub in ['clock_ma_ch', 'tpv_ma_ch', 'vol_ma_ch']:
            for c in [39, 78, 130]:
                for theta in Thetas:
                    if sub not in out_res[year][c][theta].keys(): #Check if the computation has already been done; otherwise, do it
                        q_daily_pred, es_daily_pred =\
                            models.Ch_RealizedRisk_MA(theta).fit(
                                c, out_res[year][c]['states'][sub[:-3]], jobs=2)

                        out_res[year][c][theta][sub] =\
                            {'qr':q_daily_pred, 'er':es_daily_pred}
        print('Finished CH MA')
        # Save the progress
        with open(f'{output_path}{file.split(".")[0]}.pickle', 'wb') as f:
            pickle.dump(out_res, f)

        #-------------------------- MonteCarlo Approach iid --------------------------
        for sub in ['clock_iid_mc', 'tpv_iid_mc', 'vol_iid_mc']:
            for c in [39, 78, 130]:
                if sub not in out_res[year][c][Thetas[-1]].keys(): #Check if the computation has already been done; otherwise, do it
                    q_daily_pred, es_daily_pred = models.MC_RealizedRisk(
                        Thetas).fit(
                            N_sims, c,
                            out_res[year][c]['states'][sub[:-3]],
                            ant_v=True, seed=random_seed)

                    for theta in Thetas:
                        out_res[year][c][theta][sub] =\
                            {'qr':q_daily_pred[theta], 'er':es_daily_pred[theta]}
                # Save the progress
                with open(f'{output_path}{file.split(".")[0]}.pickle', 'wb') as f:
                    pickle.dump(out_res, f)

        print('Finished MonteCarlo iid')

        #-------------------------- MonteCarlo Approach ma --------------------------
        for sub in ['clock_ma_mc', 'tpv_ma_mc', 'vol_ma_mc']:
            for c in [39, 78, 130]:
                if sub not in out_res[year][c][Thetas[-1]].keys(): #Check if the computation has already been done; otherwise, do it
                    q_daily_pred, es_daily_pred = models.MC_RealizedRisk_MA(
                        Thetas).fit(
                            N_sims, c,
                            out_res[year][c]['states'][sub[:-3]],
                            ant_v=True, seed=random_seed)

                    for theta in Thetas:
                        out_res[year][c][theta][sub] =\
                            {'qr':q_daily_pred[theta], 'er':es_daily_pred[theta]}
                # Save the progress
                with open(f'{output_path}{file.split(".")[0]}.pickle', 'wb') as f:
                    pickle.dump(out_res, f)

        print('Finished MonteCarlo ma')

        #-------------------------- Ensemble Ch-MC --------------------------
        for sub in ['clock_iid', 'tpv_iid', 'vol_iid', 'clock_ma', 'tpv_ma', 'vol_ma']:
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
                # Save the progress
                with open(f'{output_path}{file.split(".")[0]}.pickle', 'wb') as f:
                    pickle.dump(out_res, f)

        print('Finished Ensemble Ch-MC')

        print('\n')
        print('Finished year', year, '\n\n\n')
