#%% Importing the necessary libraries and preparing the environment

import utils
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import eval_utils

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

seed = 2

Algos = [
    'clock_dh', 'tpv_dh', 'vol_dh',
    'clock_iid_en', 'tpv_iid_en', 'vol_iid_en',
    'clock_ma_en', 'tpv_ma_en', 'vol_ma_en']

Cs = [39, 78, 130]
Thetas = [0.05, 0.025, 0.01]

Assets = ['BAC', 'MS', 'C', 'BK', 'STT', 'JPM', 'GS', 'WFC']
intro = 'output/bank/'

# Define the years
Years = dict()
tot_runs = 0
for asset in Assets:
    if asset not in ['MS', 'GS']:
        Years[asset] = range(1998, 2021)
    elif asset == 'MS':
        Years[asset] = range(2006, 2021)
    elif asset == 'GS':
        Years[asset] = range(2000, 2021)
    tot_runs += len(Years[asset])

# Load the predictions
out_res = dict()
for asset in Assets:
    with open(f'{intro}{asset}.pickle', 'rb') as f:
        temp_res = pickle.load(f)
    out_res[asset] = temp_res

#%% In-Sample: Hits Frequency for VaR assessment - Latex table

# Print the header of the table
print('\\begin{tabular}{lccccccccc}')
print('\\toprule')
print("\\multicolumn{10}{c}{\\textbf{BANK DATASET}} \\\\")
print('\\toprule')
print('\multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{3}{c}{$\\theta=0.05$} & \multicolumn{3}{c}{$\\theta=0.025$} & \multicolumn{3}{c}{$\\theta=0.01$} \\\\')
print('\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}')
print(' & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ \\\\')
print('\\midrule')

row2print = dict() #Initialize the dictionary to print the results

for sub in Algos:
    row2print[sub] = '\\_'.join(sub.split('_'))

for theta in Thetas:
    for c in [39, 78, 130]:
        top_q, top_loss_q = list(), np.inf
        sec_q, sec_loss_q = list(), np.inf
        print_q = dict()
        for sub in Algos:
            print_q[sub] = 0
            for asset in Assets:
                for year in Years[asset]:
                    # Compute the hits frequency
                    hits = np.array(list(
                        out_res[asset][year][c][theta][sub]['qr'].values()
                        )).flatten() >= out_res[asset][year]['y'].flatten()
                    print_q[sub] += hits.sum() / len(hits)
            print_q[sub] /= tot_runs

            # Keep track of the best and second best to highlight them in the table
            if np.abs(theta-print_q[sub]) < top_loss_q:
                sec_q, sec_loss_q = top_q, top_loss_q
                top_q, top_loss_q = [sub], np.abs(theta-print_q[sub])
            elif np.abs(theta-print_q[sub]) == top_loss_q:
                top_q.append(sub)
            elif np.abs(theta-print_q[sub]) < sec_loss_q:
                sec_q, sec_loss_q = [sub], np.abs(theta-print_q[sub])
            elif np.abs(theta-print_q[sub]) == sec_loss_q:
                sec_q.append(sub)
        
        # The top result is in bold; the second best is underlined; the rest is plain
        if len(top_q) > 1: sec_q=list()
        for sub in Algos:
            if sub in top_q:
                row2print[sub] += ' & \\textbf{' + f'{round(print_q[sub], 3):.3f}' + '}'
            elif sub in sec_q:
                row2print[sub] += ' & \\underline{' + f'{round(print_q[sub], 3):.3f}' + '}'
            else:
                row2print[sub] += ' & ' + f'{round(print_q[sub], 3):.3f}'
        
# Print the core of the table
for sub in Algos:
    print(row2print[sub] + '\\\\')
    if sub[:3] == 'vol':
        print('\\midrule')

# Print the bottom of the table
print('\\bottomrule')
print('\\end{tabular}')
print('')
print('\\vspace{0.1cm}')
print('')

#%% In-Sample: Acerbi-Szekely bootstrap test for (VaR, ES) assessment - Latex table

p_thr = 0.05 #Set the p-value threshold
as_test = eval_utils.AS14_test(one_side=False) #Initialize the Acerbi Szekely test
row2print = dict() #Initialize the dictionary to print the results

# Print the header of the table
print('\\begin{tabular}{lcccccccccccc}')
print('\\toprule')
print("\\multicolumn{13}{c}{\\textbf{BANK DATASET}} \\\\")
print('\\toprule')
print('\multirow{4}{*}{\\textbf{Algorithm}} & \multicolumn{6}{c}{$\\theta=0.05$} & \multicolumn{6}{c}{$\\theta=0.025$} \\\\')
print('\cmidrule(lr){2-7} \cmidrule(lr){8-13}')
print(' & \multicolumn{2}{c}{$c=39$} & \multicolumn{2}{c}{$c=78$} & \multicolumn{2}{c}{$c=130$} & \multicolumn{2}{c}{$c=39$} & \multicolumn{2}{c}{$c=78$} & \multicolumn{2}{c}{$c=130$} \\\\')
print('\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}')
print(' & AS1 & AS2 & AS1 & AS2 & AS1 & AS2 & AS1 & AS2 & AS1 & AS2 & AS1 & AS2 \\\\')
print('\\midrule')

for sub in Algos:
    row2print[sub] = '\\_'.join(sub.split('_'))

for theta in [0.05, 0.025]:
    for c in [39, 78, 130]:
        top_1, top_loss_1 = list(), np.inf
        sec_1, sec_loss_1 = list(), np.inf
        top_2, top_loss_2 = list(), np.inf
        sec_2, sec_loss_2 = list(), np.inf
        print_1 = dict()
        print_2 = dict()
        for sub in Algos:
            # Perform the Acerbi-Szekely test - Z1 statistic
            print_1[sub] = 0
            for asset in Assets:
                for year in Years[asset]:
                    if as_test(np.array(list(
                        out_res[asset][year][c][theta][sub]['qr'].values())),
                        np.array(list(
                            out_res[asset][year][c][theta][sub]['er'].values())),
                        out_res[asset][year]['y'],
                        test_type='Z1', theta=theta, seed=seed)['p_value'] < p_thr:
                        print_1[sub] += 1
            print_1[sub] /= tot_runs
            
            # Perform the Acerbi-Szekely test - Z2 statistic
            print_2[sub] = 0
            for asset in Assets:
                for year in Years[asset]:
                    if as_test(np.array(list(
                        out_res[asset][year][c][theta][sub]['qr'].values())),
                        np.array(list(
                            out_res[asset][year][c][theta][sub]['er'].values())),
                        out_res[asset][year]['y'],
                        test_type='Z2', theta=theta, seed=seed)['p_value'] < p_thr:
                        print_2[sub] += 1
            print_2[sub] /= tot_runs

            # Keep track of the best and second best to highlight them in the table
            if print_1[sub] < top_loss_1:
                sec_1, sec_loss_1 = top_1, top_loss_1
                top_1, top_loss_1 = [sub], print_1[sub]
            elif print_1[sub] == top_loss_1:
                top_1.append(sub)
            elif print_1[sub] < sec_loss_1:
                sec_1, sec_loss_1 = [sub], print_1[sub]
            elif print_1[sub] == sec_loss_1:
                sec_1.append(sub)

            if print_2[sub] < top_loss_2:
                sec_2, sec_loss_2 = top_2, top_loss_2
                top_2, top_loss_2 = [sub], print_2[sub]
            elif print_2[sub] == top_loss_2:
                top_2.append(sub)
            elif print_2[sub] < sec_loss_2:
                sec_2, sec_loss_2 = [sub], print_2[sub]
            elif print_2[sub] == sec_loss_2:
                sec_2.append(sub)
        
        # The top result is in bold; the second best is underlined; the rest is plain
        if len(top_1) > 1: sec_1=list()
        if len(top_2) > 1: sec_2=list()
        for sub in Algos:
            if sub in top_1:
                row2print[sub] += ' & \\textbf{' + f'{round(print_1[sub], 3):.3f}' + '}'
            elif sub in sec_1:
                row2print[sub] += ' & \\underline{' + f'{round(print_1[sub], 3):.3f}' + '}'
            else:
                row2print[sub] += ' & ' + f'{round(print_1[sub], 3):.3f}'
        for sub in Algos:
            if sub in top_2:
                row2print[sub] += ' & \\textbf{' + f'{round(print_2[sub], 3):.3f}' + '}'
            elif sub in sec_2:
                row2print[sub] += ' & \\underline{' + f'{round(print_2[sub], 3):.3f}' + '}'
            else:
                row2print[sub] += ' & ' + f'{round(print_2[sub], 3):.3f}'

# Print the core of the table
for sub in Algos:
    print(row2print[sub] + '\\\\')
    if sub[:3] == 'vol':
        print('\\midrule')

# Print the bottom of the table
print('\\bottomrule')
print('\\end{tabular}')
print('')
print('\\vspace{0.1cm}')
print('')

#%% Out-of-Sample: AR model - Compute forecasts

from statsmodels.tsa.ar_model import AutoReg

import pyximport; pyximport.install()
pyximport.install(setup_args={'include_dirs': np.get_include()})
import cython_fun

Algos = [
    'clock_dh', 'tpv_dh', 'vol_dh',
    'clock_iid_ch', 'tpv_iid_ch', 'vol_iid_ch',
    'clock_ma_ch', 'tpv_ma_ch', 'vol_ma_ch',
    'clock_iid_mc', 'tpv_iid_mc', 'vol_iid_mc',
    'clock_ma_mc', 'tpv_ma_mc', 'vol_ma_mc']

# Change the years as the first ones are used for training
Years = dict()
for asset in Assets:
    if asset not in ['MS', 'GS']:
        Years[asset] = range(2003, 2021)
    elif asset == 'MS':
        Years[asset] = range(2011, 2021)
    elif asset == 'GS':
        Years[asset] = range(2005, 2021)

out_res = dict()

for asset in Assets:
    out_res[asset] = dict()
    with open(f'{intro}{asset}.pickle', 'rb') as f:
        in_res = pickle.load(f)

    for year in Years[asset]:
        out_res[asset][year] = dict()
        for c in [39, 78, 130]:
            out_res[asset][year][c] = dict()
            for theta in Thetas:
                out_res[asset][year][c][theta] = dict()
                for sub in Algos:
                    out_res[asset][year][c][theta][sub] = dict()

                    # Extract the train set
                    q_train_set = list()
                    e_train_set = list()
                    for year_lag in range(5, 0, -1):
                        q_train_set += list(
                            in_res[
                                year-year_lag][c][theta][sub]['qr'].values())
                        e_train_set += list(
                            in_res[
                                year-year_lag][c][theta][sub]['er'].values())

                    # Extract the test set
                    q_test_set = list(
                        in_res[year][c][theta][sub]['qr'].values())
                    e_test_set = list(
                        in_res[year][c][theta][sub]['er'].values())
                    
                    # Fill NaN
                    q_train_set = cython_fun.fill_nan(
                        np.array(q_train_set),
                        np.array(q_train_set)[~np.isnan(q_train_set)][0])
                    e_train_set = cython_fun.fill_nan(
                        np.array(e_train_set),
                        np.array(e_train_set)[~np.isnan(e_train_set)][0])
                    q_test_set = cython_fun.fill_nan(
                        np.array(q_test_set), q_train_set[-1])
                    e_test_set = cython_fun.fill_nan(
                        np.array(e_test_set), e_train_set[-1])

                    # Train the AR model
                    mdl_q = AutoReg(q_train_set, lags=1).fit()
                    q_pred = cython_fun.ar1_predict_update(
                        q_train_set[-1], np.array(q_test_set), mdl_q.params)

                    mdl_e = AutoReg(e_train_set, lags=1).fit()
                    e_pred = cython_fun.ar1_predict_update(
                        e_train_set[-1], np.array(e_test_set), mdl_e.params)
                    
                    out_res[asset][year][c][theta][sub]['qf'] = q_pred
                    out_res[asset][year][c][theta][sub]['ef'] = e_pred
                
                # Ensemble
                for sub in [
                    'clock_iid', 'tpv_iid', 'vol_iid',
                    'clock_ma', 'tpv_ma', 'vol_ma' ]:
                    out_res[asset][year][c][theta][sub+'_en'] = {
                        'qf':(out_res[asset][year][c][theta][sub+'_ch']['qf'] +\
                            out_res[asset][year][c][theta][sub+'_mc']['qf'] ) / 2,
                        'ef':(out_res[asset][year][c][theta][sub+'_ch']['ef'] +\
                            out_res[asset][year][c][theta][sub+'_mc']['ef'] ) / 2 }

    # Save the progress
    with open(f'{intro}preds_AR.pickle', 'wb') as f:
        pickle.dump(out_res, f)
    print('Finished asset', asset)

#%% Out-of-Sample: EMA and RW model - Compute forecasts

from statsmodels.tsa.ar_model import AutoReg

import pyximport; pyximport.install()
pyximport.install(setup_args={'include_dirs': np.get_include()})
import cython_fun

Algos = [
    'clock_dh', 'tpv_dh', 'vol_dh',
    'clock_iid_ch', 'tpv_iid_ch', 'vol_iid_ch',
    'clock_ma_ch', 'tpv_ma_ch', 'vol_ma_ch',
    'clock_iid_mc', 'tpv_iid_mc', 'vol_iid_mc',
    'clock_ma_mc', 'tpv_ma_mc', 'vol_ma_mc']

# Change the years as the first ones are used for training
Years = dict()
for asset in Assets:
    if asset not in ['MS', 'GS']:
        Years[asset] = range(1999, 2021)
    elif asset == 'MS':
        Years[asset] = range(2007, 2021)
    elif asset == 'GS':
        Years[asset] = range(2001, 2021)

for alpha in [0.9, 1]:
    out_res = dict()

    for asset in Assets:
        out_res[asset] = dict()
        with open(f'{intro}{asset}.pickle', 'rb') as f:
            in_res = pickle.load(f)

        for year in Years[asset]:
            out_res[asset][year] = dict()
            for c in [39, 78, 130]:
                out_res[asset][year][c] = dict()
                for theta in Thetas:
                    out_res[asset][year][c][theta] = dict()
                    for sub in Algos:
                        out_res[asset][year][c][theta][sub] = dict()

                        # Extract the train set
                        q_train_set = list(
                                in_res[year-1][
                                    c][theta][sub]['qr'].values())
                        e_train_set = list(
                                in_res[year-1][
                                    c][theta][sub]['er'].values())

                        # Extract the test set
                        q_test_set = list(
                            in_res[year][c][theta][sub]['qr'].values())
                        e_test_set = list(
                            in_res[year][c][theta][sub]['er'].values())
                        
                        # Fill NaN
                        q_train_set = cython_fun.fill_nan(
                            np.array(q_train_set),
                            np.array(q_train_set)[~np.isnan(q_train_set)][0])
                        e_train_set = cython_fun.fill_nan(
                            np.array(e_train_set),
                            np.array(e_train_set)[~np.isnan(e_train_set)][0])
                        q_test_set = cython_fun.fill_nan(
                            np.array(q_test_set), q_train_set[-1])
                        e_test_set = cython_fun.fill_nan(
                            np.array(e_test_set), e_train_set[-1])

                        # Predict with ESWA
                        out_res[asset][year][c][theta][sub]['qf'] =\
                            cython_fun.ESWA(q_train_set, q_test_set, alpha)
                        out_res[asset][year][c][theta][sub]['ef'] =\
                            cython_fun.ESWA(e_train_set, e_test_set, alpha)
                        
                    # Ensemble
                    for sub in [
                        'clock_iid', 'tpv_iid', 'vol_iid',
                        'clock_ma', 'tpv_ma', 'vol_ma' ]:
                        out_res[asset][year][c][theta][sub+'_en'] = {
                            'qf':(out_res[asset][year][c][theta][sub+'_ch']['qf'] +\
                                    out_res[asset][year][c][theta][sub+'_mc']['qf'] ) / 2,
                            'ef':(out_res[asset][year][c][theta][sub+'_ch']['ef'] +\
                                    out_res[asset][year][c][theta][sub+'_mc']['ef'] ) / 2
                            }

        # Save the progress
        if alpha < 1:
            with open(f'{intro}preds_EMA.pickle', 'wb') as f:
                pickle.dump(out_res, f)
        else:
            with open(f'{intro}preds_RW.pickle', 'wb') as f:
                pickle.dump(out_res, f)

#%% Out-of-Sample - Latex table

Algos2Print = [
    'clock_dh', 'tpv_dh', 'vol_dh',
    'clock_iid_en', 'tpv_iid_en', 'vol_iid_en',
    'clock_ma_en', 'tpv_ma_en', 'vol_ma_en']

for predictor in ['AR', 'EMA', 'RW']:
    print('Working with predictor', predictor, '\n')
    with open(f'{intro}preds_{predictor}.pickle', 'rb') as f:
        out_res = pickle.load(f)
    
    if predictor == 'AR':
        Years = dict()
        tot_runs = 0
        for asset in Assets:
            if asset not in ['MS', 'GS']:
                Years[asset] = range(2003, 2021)
            elif asset == 'MS':
                Years[asset] = range(2011, 2021)
            elif asset == 'GS':
                Years[asset] = range(2005, 2021)
                tot_runs += len(Years[asset])
    else:
        Years = dict()
        tot_runs = 0
        for asset in Assets:
            if asset not in ['MS', 'GS']:
                Years[asset] = range(1999, 2021)
            elif asset == 'MS':
                Years[asset] = range(2007, 2021)
            elif asset == 'GS':
                Years[asset] = range(2001, 2021)
        tot_runs += len(Years[asset])

    #-------------------- Step 1: VaR Table
    # Print the header of the table
    print('\\begin{tabular}{lccccccccc}')
    print('\\toprule')
    print("\\multicolumn{10}{c}{\\textbf{BANK DATASET}} \\\\")
    print('\\toprule')
    print('\multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{3}{c}{$\\theta=0.05$} & \multicolumn{3}{c}{$\\theta=0.025$} & \multicolumn{3}{c}{$\\theta=0.01$} \\\\')
    print('\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}')
    print(' & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ \\\\')
    print('\\midrule')

    row2print = dict()

    for sub in Algos2Print:
        row2print[sub] = '\\_'.join(sub.split('_'))

    for theta in Thetas:
        Q_Loss = eval_utils.PinballLoss(theta)
        for c in [39, 78, 130]:
            top_q, top_loss_q = list(), np.inf
            sec_q, sec_loss_q = list(), np.inf
            print_q = dict()
            for sub in Algos2Print:
                print_q[sub] = 0
                for asset in Assets:
                    with open(f'{intro}{asset}.pickle', 'rb') as f:
                        in_res = pickle.load(f)
                    for year in Years[asset]:
                        print_q[sub] += Q_Loss(
                            out_res[asset][year][c][theta][sub]['qf'],
                            in_res[year]['y'] )
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
            for sub in Algos2Print:
                if sub in top_q:
                    row2print[sub] += ' & \\textbf{' + str(round(1e3*print_q[sub], 3)) + '}'
                elif sub in sec_q:
                    row2print[sub] += ' & \\underline{' + str(round(1e3*print_q[sub], 3)) + '}'
                else:
                    row2print[sub] += ' & ' + str(round(1e3*print_q[sub], 3))
            
    # Print the core of the table
    for sub in Algos2Print:
        print(row2print[sub] + '\\\\')
        if sub[:3] == 'vol':
            print('\\midrule')

    # Print the bottom of the table
    print('\\bottomrule')
    print('\\end{tabular}')
    print('')
    print('\\vspace{0.1cm}')
    print('')

    #-------------------- Step 2: ES Table

    # Print the header of the table
    print('\\begin{tabular}{lcccccc}')
    print('\\toprule')
    print(" & \multicolumn{6}{c}{BANK DATASET} \\\\")
    print('\cmidrule(lr){2-7}')
    print('\multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{3}{c}{$\\theta=0.05$} & \multicolumn{3}{c}{$\\theta=0.025$} \\\\')
    print('\cmidrule(lr){2-4} \cmidrule(lr){5-7}')
    print(' & $c=39$ & $c=78$ & $c=130$ & $c=39$ & $c=78$ & $c=130$ \\\\')
    print('\\midrule')

    row2print = dict()

    for sub in Algos2Print:
        row2print[sub] = '\\_'.join(sub.split('_'))

    for theta in [0.05, 0.025]:
        ES_Loss = eval_utils.patton_loss(theta)
        for c in [39, 78, 130]:
            top_q, top_loss_q = list(), np.inf
            sec_q, sec_loss_q = list(), np.inf
            print_q = dict()
            for sub in Algos2Print:
                print_q[sub] = 0
                for asset in Assets:
                    with open(f'{intro}{asset}.pickle', 'rb') as f:
                        in_res = pickle.load(f)
                    for year in Years[asset]:
                        print_q[sub] += ES_Loss(
                            out_res[asset][year][c][theta][sub]['qf'],
                            out_res[asset][year][c][theta][sub]['ef'],
                            in_res[year]['y'] )
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
            for sub in Algos2Print:
                if sub in top_q:
                    row2print[sub] += ' & \\textbf{' + str(round(print_q[sub], 3)) + '}'
                elif sub in sec_q:
                    row2print[sub] += ' & \\underline{' + str(round(print_q[sub], 3)) + '}'
                else:
                    row2print[sub] += ' & ' + str(round(print_q[sub], 3))
            
    # Print the core of the table
    for sub in Algos2Print:
        print(row2print[sub] + '\\\\')
        if sub[:3] == 'vol':
            print('\\midrule')

    # Print the bottom of the table
    print('\\bottomrule')
    print('\\end{tabular}')
    print('')

    print('\nEnd predictor', predictor)

# %%
