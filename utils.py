
import numpy as np
import pandas as pd

def IntraSeries_2_DayReturn(series):
    '''
    Computes the day return of a time series representing intraday prices.

    INPUTS:
        - series: pandas Series,
        the time series of intraday prices

    OUTPUTS:
        - output: numpy array,
        the daily log returns

    Example of usage
    ----------------
    .. code-block:: python

        import pandas as pd
        from utils import IntraSeries_2_DayReturn

        df = pd.read_csv('stock.csv')
        df.index = pd.to_datetime(df.index)

        daily_ret = IntraSeries_2_DayReturn(df.Close) #Compute the daily log returns
    '''

    output = list() #Initialize the output
    for day, _ in series.groupby(pd.Grouper(freq='D')): #Iterate over each day
        end_day = day + pd.Timedelta(days=1)
        temp_y = series[(series.index >= day) & (series.index < end_day)] #Isolate the daily time series
        if len(temp_y) > 0: #If the daily time series is not empty, compute the log return
            output.append(np.log(temp_y[-1]) - np.log(temp_y[0]))
            
    output = np.array(output)
    return output
 
class Subordinator():
    '''
    Subordinator class. It is intended to be used on each day separately.

    Parameters:
    ----------------
        - c: int
            the number of time indexes to sample
        - sub_type: str, optional
            Type of subordinator to use. Either 'clock' for the clock time; 'tpv' for the TriPower Variation;
            'vol' for the volume; or 'identity' for the identity (that is, all the indexes are returned). Default is 'clock'

    Example of usage
    ----------------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from utils import subordinator

        df = pd.read_csv('stock.csv')
        df.index = pd.to_datetime(df.index)
        day_price = df.Close
        day_vol = df.Volume

        subordinator = Subordinator(78, sub_type='vol') #Initialize the subordinator

        target_idxs = day_price.iloc[subordinator.predict(day_price, vol=day_vol)] #Extract the subordinated indexes
        log_ret = np.log(target_points).diff().dropna() #Compute the subordinated logarithmic returns
    
    Methods:
    ----------------
    '''
    def __init__(self, c, sub_type='clock'):
        self.c = c+1
        self.sub_type = sub_type
    
    def sample_with_tie_break(self, a):
        '''
        Ensure the sampled time indexes are unique (i.e., not overlapping). That is, it manage the tie-break rule for transofrming a non-injective function into an injective subordinator.

        INPUTS:
            - a: ndarray
                the cumulative intensities

        OUTPUTS:
            - indices: ndarray
                the indexes corresponding to the subordinated values
        
        :meta private:
        '''
        indices = np.searchsorted(
            a, np.linspace(0, a[-1], self.c-1), side='left') # Initialize an attempt of indices
        
        for i in range(1, len(indices)): #Avoid duplicates - forward pass
            if indices[i] == indices[i - 1]:
                indices[i] += 1

        if indices[-1] >= len(a)-1: #Remove values outside the data index
            indices[-1] = len(a) - 2

        for i in range(len(indices)-1, 1, -1): #Avoid all the repretitions - backward pass
            if indices[i] <= indices[i - 1]:
                indices[i - 1] = indices[i] - 1

        return indices
    
    def predict(self, data, vol=None, tpv_int_min=15):
        '''
        Returns the index position of the subordinated values. That is, return the vector \tau(j), with j=0,..,c, where \tau is intended to be the subordinator.

        INPUTS:
            - data: pd.Series
                the time series of intra-day prices, over all the day (that is, from 09:30 to 16:00)
            - tpv_int_min: int, optional
                half-length of the window used for computing the tri-power variation. Only used when self.sub_type == 'tpv'. Default is 15
            - vol: pd.Series, optional
                the volume series, on the same indexes as data. Only used when self.sub_type == 'vol'

        OUTPUTS:
            - sub_idxs: list
                the indexes corresponding to the subordinated values
        '''
        from datetime import datetime

        # Distinguish different situations according to self.sub_type
        if self.sub_type == 'clock':
            sub_idxs = [int(
                len(data.index) / (self.c-1) * i
                ) for i in range(self.c-1)] + [len(data.index)-1]
            return sub_idxs

        elif self.sub_type == 'identity':
            sub_idxs = range(len(data.index))
            return sub_idxs
        
        elif self.sub_type == 'tpv':
            
            cte = 1.9357924048803463 #Normalization constant

            # Compute the TriPower Variation intensity
            tpv = list()
            for t in data.index: #Iterate over every minute
                temp_data = data[(data.index>=t-pd.Timedelta(minutes=tpv_int_min)) &\
                                 (data.index<=t+pd.Timedelta(minutes=tpv_int_min))].values #Isolate the time window
                temp_data = temp_data[1:] - temp_data[:-1] #Compute the price delta inside each window
                temp_data = temp_data[ temp_data != 0 ] #Remove zeros
                if len(temp_data) < 3: #If there are not enough values for computing the tpv, just return 0
                    tpv.append(0)
                else:
                    temp_tpv = 0 #Compute the sum for the tpv
                    for j in range(2, len(temp_data)):
                        temp_tpv += np.abs(temp_data[j-2])**(2/3) *\
                            np.abs(temp_data[j-1])**(2/3) *\
                                np.abs(temp_data[j])**(2/3)
                    tpv.append(temp_tpv * cte) #Normalize

            tpv = np.array(tpv).cumsum() #From intensities to cumulative intensities

            # Compute the subordinator as equispaced points in the TPV
            sub_idxs = list(self.sample_with_tie_break(tpv)) + [len(data.index)-1]
            return sub_idxs
        
        elif self.sub_type == 'vol':
            t_vol = np.where(vol.values>=0, vol.values, 0).cumsum() #Clean potential missing/corrupted values
            # Compute the subordinator as equispaced points in the vol
            sub_idxs = list(self.sample_with_tie_break(t_vol)) + [len(data.index)-1]
            return sub_idxs

def Fill_RTH_Minutes(df):
    '''
    Data preprocessing. Given a pandas dataframe, it fills missing values in the Regular Trading Hours (RTH) (09:30 - 16:00).
    Only the days where there is at least one observation are used, as they are assumed to be all and only the working days.

    INPUTS:
        - df: pandas DataFrame,
            the dataframe, made up of minute-by-minute values

    OUTPUTS:
        - df: pandas DataFrame,
            the preprocessed dataframe, with no holes and filled missing values.

    Example of usage
    ----------------
    .. code-block:: python

        import pandas as pd
        from utils import Fill_RTH_Minutes

        df = pd.read_csv('stock.csv')
        df.index = pd.to_datetime(df.index)

        df_full = Fill_RTH_Minutes(df) #Data Preprocessing
    '''
    full_days = pd.to_datetime(df.index.date).unique() #Obtain the list of days
    rth_minutes = pd.date_range('09:30', '16:00', freq='T').time # Create a time range for all minutes during RTH
    full_index = pd.MultiIndex.from_product([full_days, rth_minutes], names=['Day', 'Minute']) # Create a MultiIndex for all possible day-minute combinations

    # Reset the index of df to 'Day' and 'Minute' if it's not already
    df['Day'] = df.index.date
    df['Minute'] = df.index.time
    df.set_index(['Day', 'Minute'], inplace=True)
    # Reindex the DataFrame to include all possible minutes
    df = df.reindex(full_index)

    
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].fillna(method='ffill') # Forward fill the price columns
    df['Volume'] = df['Volume'].fillna(0) # Set Volume to 0 for newly filled minutes
    df.reset_index(inplace=True) # Reset index to have 'Day' and 'Minute' as columns if needed
    
    df['timestamp'] = pd.to_datetime(df['Day'].astype(str) + ' ' + df['Minute'].astype(str))
    df.set_index('timestamp', inplace=True)

    df.drop(columns=['Day', 'Minute'], inplace=True) # Drop 'Day' and 'Minute' columns if not needed anymore

    return df

def price2params(y, c, mu_prior=0., sub_type='clock', vol=None, ma=False, hist_mean=None):
    '''
    Fit the intra-day distribution, which is assumed to be a Student's t-distribution

    INPUTS:
        - y: pandas Series,
            the price time series, with a minute-by-minute granularity, over all the considered period (e.g., one year of data)
        - c: int
            the number of time indexes to sample
        - mu_prior: float, optional
            the prior of the intra-day mean.
            If mu_prior is > 0, the prior is EWMA with smoothing parameter equals to mu_prior.
            If mu_prior is 0, then the prior mean = constant = 0 is used.
            If mu_prior is None, then no prior is used and mu is fitted to the data, as the other distribution parameters.
            Default is 0
        - sub_type: str, optional
            Type of subordinator to use. Either 'clock' for the clock time; 'tpv' for the TriPower Variation;
            'vol' for the volume; or 'identity' for the identity (that is, all the indexes are returned). Default is 'clock'
        - vol: pd.Series, optional
            the volume series, on the same indexes as data. Only used when sub_type == 'vol'
        - ma: bool, optional
            Whether the returns are assumed to follow an MA(1) process. If False, returns are assumed to be iid. Default is False.
        - hist_mean: bool, optional
            Historical mean to use as starting point for computing the EWMA prior of the mean. Only needed when mu_prior > 0.

    OUTPUTS:
        - out_pars: dict,
            the fitted parameters. Every key correspond to a day of the y series. Every value is a list containing nu, mu, sigma.

    Example of usage
    ----------------
    .. code-block:: python

        import pandas as pd
        from utils import price2params

        df = pd.read_csv('stock.csv')
        price = df.Close
        vol = df.Volume

        fitted_pars = price2params(price, mu_prior=1/3, c=78, sub_type='vol', vol=vol, hist_mean=0)
    '''
    if isinstance(mu_prior, type(None)):
        if ma:
            return _price2params_ma(y, c, mu=mu_prior, sub_type=sub_type, vol=vol)
        else:
            return _price2params_iid(y, c, mu=mu_prior, sub_type=sub_type, vol=vol)
    else:
        if mu_prior > 0:
            if ma:
                return _price2params_ma_EWMA(
                    y, c, mu=hist_mean, smooth_par=mu_prior, sub_type=sub_type, vol=vol)
            else:
                return _price2params_iid_EWMA(
                    y, c, mu=hist_mean, smooth_par=mu_prior, sub_type=sub_type, vol=vol)
        else:
            if ma:
                return _price2params_ma(y, c, mu=mu_prior, sub_type=sub_type, vol=vol)
            else:
                return _price2params_iid(y, c, mu=mu_prior, sub_type=sub_type, vol=vol)
        

def _price2params_iid(y, c, mu=0., sub_type='clock', vol=None):
    '''
    Fit the intra-day distribution, which is assumed to be a Student's t-distribution

    INPUTS:
        - y: pandas Series,
            the price time series, with a minute-by-minute granularity, over all the considered period (e.g., one year of data)
        - c: int
            the number of time indexes to sample
        - mu: float,
            the intra-day mean. It could be either a float or None. In the latter case, it will be estimated from the data. It is preferable to not set mu=None.
            If you don't have a reliable estimate for it, simply use 0. Default is 0
        - sub_type: str, optional
            Type of subordinator to use. Either 'clock' for the clock time; 'tpv' for the TriPower Variation;
            'vol' for the volume; or 'identity' for the identity (that is, all the indexes are returned). Default is 'clock'
        - vol: pd.Series, optional
            the volume series, on the same indexes as data. Only used when sub_type == 'vol'

    OUTPUTS:
        - out_pars: dict,
            the fitted parameters. Every key correspond to a day of the y series. Every value is a list containing nu, mu, sigma.

    Example of usage
    ----------------
    .. code-block:: python

        import pandas as pd
        from utils import price2params

        df = pd.read_csv('stock.csv')
        price = df.Close
        vol = df.Volume

        fitted_pars = price2params(price, c=78, sub_type='vol', vol=vol)

    :meta private:
    '''
    from scipy.stats import t as stud_t #Load the Student's t-distribution

    subordinator = Subordinator(c, sub_type=sub_type) #Initialize the subordinator

    out_pars = dict() #Initialize the output dictionary
    for day, _ in y.groupby(pd.Grouper(freq='D')): #Iterate over the days
        # Isolate the day to work with
        end_day = day + pd.Timedelta(days=1)
        temp_y = y[(y.index >= day) & (y.index < end_day)]
        #Eventually, ass the volume
        if isinstance(vol, type(None)):
            temp_vol = None
        else:
            temp_vol = vol[(vol.index >= day) & (vol.index < end_day)]

        if len(temp_y) > 0: #If it is a working day
            target_points = temp_y.iloc[subordinator.predict(temp_y, vol=temp_vol)] #Subordinated prices
            log_ret = np.log(target_points).diff().dropna() # Subordinated logarithmic returns

            if isinstance(mu, type(None)): #If mu is not given, it has to be estimated
                temp_out = list(stud_t.fit(log_ret))
                if temp_out[0] < 2+1e-6: #Cap on the degrees of freedom nu
                    temp_out = list(stud_t.fit(log_ret, f0=2+1e-6))
                    if temp_out[2] < 1e-6: #Cap on the standard deviation
                        temp_out[2] = 1e-6
            else: #If mu is given, fit the other parameters by keeping it fixed
                temp_out = list(stud_t.fit(log_ret, floc=mu))
                if temp_out[0] < 2+1e-6: #Cap on the degrees of freedom nu
                    temp_out = list(stud_t.fit(log_ret, f0=2+1e-6, floc=mu))
                    if temp_out[2] < 1e-6: #Cap on the standard deviation
                        temp_out[2] = 1e-6

            out_pars[day] = temp_out #Add the results to the output dictionary

    return out_pars

def _price2params_iid_EWMA(y, c, mu=0., smooth_par=0.9, sub_type='clock', vol=None):
    '''
    Fit the intra-day distribution, which is assumed to be a Student's t-distribution.
    The mean is updated according to an EWMA.

    INPUTS:
        - y: pandas Series,
            the price time series, with a minute-by-minute granularity, over all the considered period (e.g., one year of data)
        - c: int
            the number of time indexes to sample
        - mu: float, optional
            the intra-day mean. It could be either a float or None. In the latter case, it will be estimated from the data. It is preferable to not set mu=None.
            If you don't have a reliable estimate for it, simply use 0. Default is 0
        - smooth_par: float, optional
            the smoothing parameter for the EWMA. Default is 0.9
        - sub_type: str, optional
            Type of subordinator to use. Either 'clock' for the clock time; 'tpv' for the TriPower Variation;
            'vol' for the volume; or 'identity' for the identity (that is, all the indexes are returned). Default is 'clock'
        - vol: pd.Series, optional
            the volume series, on the same indexes as data. Only used when sub_type == 'vol'

    OUTPUTS:
        - out_pars: dict,
            the fitted parameters. Every key correspond to a day of the y series. Every value is a list containing nu, mu, sigma.

    Example of usage
    ----------------
    .. code-block:: python

        import pandas as pd
        from utils import price2params_EWMA

        df = pd.read_csv('stock.csv')
        price = df.Close
        vol = df.Volume

        fitted_pars = price2params_EWMA(price, c=78, sub_type='vol', vol=vol)

    :meta private:
    '''
    from scipy.stats import t as stud_t #Load the Student's t-distribution

    subordinator = Subordinator(c, sub_type=sub_type) #Initialize the subordinator

    out_pars = dict() #Initialize the output dictionary
    for day, _ in y.groupby(pd.Grouper(freq='D')): #Iterate over the days
        # Isolate the day to work with
        end_day = day + pd.Timedelta(days=1)
        temp_y = y[(y.index >= day) & (y.index < end_day)]
        #Eventually, ass the volume
        if isinstance(vol, type(None)):
            temp_vol = None
        else:
            temp_vol = vol[(vol.index >= day) & (vol.index < end_day)]

        if len(temp_y) > 0: #If it is a working day
            target_points = temp_y.iloc[subordinator.predict(temp_y, vol=temp_vol)] #Subordinated prices
            log_ret = np.log(target_points).diff().dropna() # Subordinated logarithmic returns
            '''
            mu = smooth_par * (
                np.log(target_points[-1]) - np.log(target_points[0])
                )/c + (1 - smooth_par) * mu #Update the EWMA
            '''
            if isinstance(mu, type(None)): #If mu is not given, it has to be estimated
                temp_out = list(stud_t.fit(log_ret))
                if temp_out[0] < 2+1e-6: #Cap on the degrees of freedom nu
                    temp_out = list(stud_t.fit(log_ret, f0=2+1e-6))
                    if temp_out[2] < 1e-6: #Cap on the standard deviation
                        temp_out[2] = 1e-6
            else: #If mu is given, fit the other parameters by keeping it fixed
                temp_out = list(stud_t.fit(log_ret, floc=mu))
                if temp_out[0] < 2+1e-6: #Cap on the degrees of freedom nu
                    temp_out = list(stud_t.fit(log_ret, f0=2+1e-6, floc=mu))
                    if temp_out[2] < 1e-6: #Cap on the standard deviation
                        temp_out[2] = 1e-6

            out_pars[day] = temp_out #Add the results to the output dictionary

            mu = smooth_par * (
                np.log(target_points[-1]) - np.log(target_points[0])
                )/c + (1 - smooth_par) * mu #Update the EWMA

    return out_pars

def _price2params_ma(y, c, mu=0., sub_type='clock', vol=None):
    '''
    Fit the intra-day distribution, which is assumed to be a MA(1) process with Student's t innovations.

    INPUTS:
        - y: pandas Series,
            the price time series, with a minute-by-minute granularity, over all the considered period (e.g., one year of data)
        - c: int
            the number of time indexes to sample
        - mu: float, optional
            the intra-day mean. It could be either a float or None. In the latter case, it will be estimated from the data. It is preferable to not set mu=None.
            If you don't have a reliable estimate for it, simply use 0. Default is 0
        - sub_type: str, optional
            Type of subordinator to use. Either 'clock' for the clock time; 'tpv' for the TriPower Variation;
            'vol' for the volume; or 'identity' for the identity (that is, all the indexes are returned). Default is 'clock'
        - vol: pd.Series, optional
            the volume series, on the same indexes as data. Only used when sub_type == 'vol'

    OUTPUTS:
        - out_pars: dict,
            the fitted parameters. Every key correspond to a day of the y series. Every value is a list containing nu, mu, sigma.

    Example of usage
    ----------------
    .. code-block:: python

        import pandas as pd
        from utils import price2params_ma

        df = pd.read_csv('stock.csv')
        price = df.Close
        vol = df.Volume

        fitted_pars = price2params_ma(price, c=39, sub_type='vol', vol=vol)

    :meta private:
    '''
    from scipy.stats import t as stud_t #Load the Student's t-distribution
    from scipy.optimize import minimize, Bounds #Load function for minimization

    def my_t_fit(obs, mu): #Fit the innovations to a Student's t-distribution
        if isinstance(mu, type(None)): #If mu is not given, it has to be estimated
            temp_out = list(stud_t.fit(obs))
            if temp_out[0] < 2+1e-6: #Cap on the degrees of freedom nu
                temp_out = list(stud_t.fit(obs, f0=2+1e-6))
                if temp_out[2] < 1e-6: #Cap on the standard deviation
                    temp_out[2] = 1e-6
        else: #If mu is given, fit the other parameters by keeping it fixed
            temp_out = list(stud_t.fit(obs, floc=mu))
            if temp_out[0] < 2+1e-6: #Cap on the degrees of freedom nu
                temp_out = list(stud_t.fit(obs, f0=2+1e-6, floc=mu))
                if temp_out[2] < 1e-6: #Cap on the standard deviation
                    temp_out[2] = 1e-6
        return temp_out
    
    def ma_resids(y, phi, xi0): #Filter the residuals of the MA(1) process
        inn = [xi0] #Initialize the residuals
        for y_curr in y: #Iterate over the observations
            inn.append( y_curr - phi*inn[-1] )
        return inn

    def ma_neg_ll(params, y, mu): #Negative log-likelihood of the MA(1) process
        phi, xi0 = params
        inn = ma_resids(y, phi, xi0) #Filter the residuals
        if isinstance(mu, type(None)): #If mu is not given, it has to be estimated
            return - stud_t.logpdf(inn, *my_t_fit(inn, None)).sum() #Return the negative log-likelihood
        else: #If mu is given, it has to been adjusted according to phi
            return - stud_t.logpdf(inn, *my_t_fit(inn, mu/(1+phi))).sum() #Return the negative log-likelihood

    starting_point = [-0.05, 0] #Starting point for the minimization: phi=-0.05, xi0=0
    bounds = Bounds([-0.1, -1], [0.1, 1]) #Bounds for the parameters

    subordinator = Subordinator(c, sub_type=sub_type) #Initialize the subordinator

    out_pars = dict() #Initialize the output dictionary
    for day, _ in y.groupby(pd.Grouper(freq='D')): #Iterate over the days
        # Isolate the day to work with
        end_day = day + pd.Timedelta(days=1)
        temp_y = y[(y.index >= day) & (y.index < end_day)]
        #Eventually, ass the volume
        if isinstance(vol, type(None)):
            temp_vol = None
        else:
            temp_vol = vol[(vol.index >= day) & (vol.index < end_day)]

        if len(temp_y) > 0: #If it is a working day
            target_points = temp_y.iloc[subordinator.predict(temp_y, vol=temp_vol)] #Subordinated prices
            log_ret = np.log(target_points).diff().dropna() # Subordinated logarithmic returns

            temp_out = minimize(
                ma_neg_ll, starting_point, args=(log_ret, mu),
                method='SLSQP', bounds=bounds).x #Find the optimal phi and starting point
            temp_out = [temp_out[0]] + list(
                my_t_fit(ma_resids(log_ret, *temp_out), mu/(1+temp_out[0]))) #Concatenate phi and innovations' parameters
            out_pars[day] = temp_out #Add the results to the output dictionary

    return out_pars

def _price2params_ma_EWMA(y, c, mu=0., smooth_par=0.9, sub_type='clock', vol=None):
    '''
    Fit the intra-day distribution, which is assumed to be a MA(1) process with Student's t innovations.
    The mean is updated according to an EWMA.

    INPUTS:
        - y: pandas Series,
            the price time series, with a minute-by-minute granularity, over all the considered period (e.g., one year of data)
        - c: int
            the number of time indexes to sample
        - mu: float, optional
            the intra-day mean. It could be either a float or None. In the latter case, it will be estimated from the data. It is preferable to not set mu=None.
            If you don't have a reliable estimate for it, simply use 0. Default is 0
        - smooth_par: float, optional
            the smoothing parameter for the EWMA. Default is 0.9
        - sub_type: str, optional
            Type of subordinator to use. Either 'clock' for the clock time; 'tpv' for the TriPower Variation;
            'vol' for the volume; or 'identity' for the identity (that is, all the indexes are returned). Default is 'clock'
        - vol: pd.Series, optional
            the volume series, on the same indexes as data. Only used when sub_type == 'vol'

    OUTPUTS:
        - out_pars: dict,
            the fitted parameters. Every key correspond to a day of the y series. Every value is a list containing nu, mu, sigma.

    Example of usage
    ----------------
    .. code-block:: python

        import pandas as pd
        from utils import price2params_ma

        df = pd.read_csv('stock.csv')
        price = df.Close
        vol = df.Volume

        fitted_pars = price2params_ma(price, c=39, sub_type='vol', vol=vol)

    :meta private:
    '''
    from scipy.stats import t as stud_t #Load the Student's t-distribution
    from scipy.optimize import minimize, Bounds #Load function for minimization

    def my_t_fit(obs, mu): #Fit the innovations to a Student's t-distribution
        if isinstance(mu, type(None)): #If mu is not given, it has to be estimated
            temp_out = list(stud_t.fit(obs))
            if temp_out[0] < 2+1e-6: #Cap on the degrees of freedom nu
                temp_out = list(stud_t.fit(obs, f0=2+1e-6))
                if temp_out[2] < 1e-6: #Cap on the standard deviation
                    temp_out[2] = 1e-6
        else: #If mu is given, fit the other parameters by keeping it fixed
            temp_out = list(stud_t.fit(obs, floc=mu))
            if temp_out[0] < 2+1e-6: #Cap on the degrees of freedom nu
                temp_out = list(stud_t.fit(obs, f0=2+1e-6, floc=mu))
                if temp_out[2] < 1e-6: #Cap on the standard deviation
                    temp_out[2] = 1e-6
        return temp_out
    
    def ma_resids(y, phi, xi0): #Filter the residuals of the MA(1) process
        inn = [xi0] #Initialize the residuals
        for y_curr in y: #Iterate over the observations
            inn.append( y_curr - phi*inn[-1] )
        return inn

    def ma_neg_ll(params, y, mu): #Negative log-likelihood of the MA(1) process
        phi, xi0 = params
        inn = ma_resids(y, phi, xi0) #Filter the residuals
        if isinstance(mu, type(None)): #If mu is not given, it has to be estimated
            return - stud_t.logpdf(inn, *my_t_fit(inn, None)).sum() #Return the negative log-likelihood
        else: #If mu is given, it has to been adjusted according to phi
            return - stud_t.logpdf(inn, *my_t_fit(inn, mu/(1+phi))).sum() #Return the negative log-likelihood

    starting_point = [-0.05, 0] #Starting point for the minimization: phi=-0.05, xi0=0
    bounds = Bounds([-0.1, -1], [0.1, 1]) #Bounds for the parameters

    subordinator = Subordinator(c, sub_type=sub_type) #Initialize the subordinator

    out_pars = dict() #Initialize the output dictionary
    for day, _ in y.groupby(pd.Grouper(freq='D')): #Iterate over the days
        # Isolate the day to work with
        end_day = day + pd.Timedelta(days=1)
        temp_y = y[(y.index >= day) & (y.index < end_day)]
        #Eventually, ass the volume
        if isinstance(vol, type(None)):
            temp_vol = None
        else:
            temp_vol = vol[(vol.index >= day) & (vol.index < end_day)]

        if len(temp_y) > 0: #If it is a working day
            target_points = temp_y.iloc[subordinator.predict(temp_y, vol=temp_vol)] #Subordinated prices
            log_ret = np.log(target_points).diff().dropna() # Subordinated logarithmic returns
            '''
            mu = smooth_par * (
                np.log(target_points[-1]) - np.log(target_points[0])
                )/c + (1 - smooth_par) * mu #Update the EWMA
            '''
            temp_out = minimize(
                ma_neg_ll, starting_point, args=(log_ret, mu),
                method='SLSQP', bounds=bounds).x #Find the optimal phi and starting point
            temp_out = [temp_out[0]] + list(
                my_t_fit(ma_resids(log_ret, *temp_out), mu/(1+temp_out[0]))) #Concatenate phi and innovations' parameters
            out_pars[day] = temp_out #Add the results to the output dictionary

            mu = smooth_par * (
                np.log(target_points[-1]) - np.log(target_points[0])
                )/c + (1 - smooth_par) * mu #Update the EWMA

    return out_pars
