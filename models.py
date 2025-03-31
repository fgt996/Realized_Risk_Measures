
import utils
import numpy as np
import pandas as pd

class DH_RealizedRisk():
    '''
    Filtering approach proposed in [1]. It is based on the assumption of self-similarity.

    [1]: Dimitriadis, T., & Halbleib, R. (2022). Realized quantiles. Journal of Business & Economic Statistics, 40(3), 1346-1361.

    Parameters:
    ----------------
        - c: int
            the number of time indexes to sample.
        - theta: float
            the target probability level.
        - H: float, optional
            Hurst exponent of the subordinated log-return process. Default is 0.5.
        - sub_type: str, optional
            Type of subordinator to use. Either 'clock' for the clock time; 'tpv' for the TriPower Variation;
            'vol' for the volume; or 'identity' for the identity (that is, all the indexes are returned). Default is 'clock'

    Example of usage
    ----------------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from models import DH_RealizedRisk

        df = pd.read_csv('stock.csv')
        df.index = pd.to_datetime(df.index)
        day_price = df.Close

        mdl = DH_RealizedRisk(78, 0.05, sub_type='tpv') #Initialize the model
        qf, ef = mdl.fit(day_price) #Run to obtain the filtered VaR and ES
    
    Methods:
    ----------------
    '''
    def __init__(self, c, theta, H=0.5, sub_type='clock'):
        self.c = c
        self.H = H
        self.coeff = self.c**self.H #Set the scaling coefficient
        self.theta = theta
        self.subordinator = utils.Subordinator(c, sub_type=sub_type)
    
    def fit(self, y, vol=None, tpv_int_min=15):
        '''
        Returns the filtered quantile and expected shortfall.

        INPUTS:
            - y: pd.Series
                the time series of intra-day prices, over all the day (that is, from 09:30 to 16:00)
            - vol: pd.Series, optional
                the volume series, on the same indexes as data. Only used when self.sub_type == 'vol'
            - tpv_int_min: int, optional
                half-length of the window used for computing the tri-power variation. Only used when self.sub_type == 'tpv'. Default is 15

        OUTPUTS:
            - qf: dict
                the filtered VaR. Every key correspond to a day of the y series.
            - ef: dict
                the filtered ES. Every key correspond to a day of the y series.
        '''

        qf, ef = dict(), dict() #Initialize the dictionaries to store the daily values
        for day, _ in y.groupby(pd.Grouper(freq='D')): #Loop over all the days
            end_day = day + pd.Timedelta(days=1)
            temp_y = y[(y.index >= day) & (y.index < end_day)] #Select the data for the day
            if isinstance(vol, type(None)): #If no volume is provided, set it to None
                temp_vol = None
            else: #Otherwise, select the volume for the day
                temp_vol = vol[(vol.index >= day) & (vol.index < end_day)]

            if len(temp_y) > 0: #If it is a working day
                target_points = temp_y.iloc[self.subordinator.predict(
                    temp_y, vol=temp_vol, tpv_int_min=tpv_int_min)]  #Subordinated prices
                log_ret = np.log(target_points).diff().dropna() # Subordinated logarithmic returns

                q_emp = np.quantile(log_ret, self.theta) #Empirical quantile
                es_emp = np.mean(log_ret[log_ret <= q_emp]) #Empirical expected shortfall
                
                qf[day] = q_emp * self.coeff
                ef[day] = es_emp * self.coeff

        return qf, ef

def _custom_brentq(target_fun, a, x0, tol=1e-8, max_it=10):
    '''
    Customized routine for finding zeros of the "CDF-theta" function.

    INPUTS:
        - target_fun: callable,
            the target function, which is assumed to have a root in the interval [a, 0].
        - a: float,
            the left bound for the searching domain.
        - x0: float,
            the starting point for the search
        - tol: float, optional
            the required tolerance (on |target_fun|). Default is 1e-8.
        - max_it: int, optional
            the maximum number of iterations of the brentq method. Default is 10.

    OUTPUTS:
        - zero: float,
            the root of the function.
    '''
    from scipy.optimize import brentq #Load the brentq function

    # Check if the starting point is a root
    x0_val = target_fun(x0)
    if x0_val == 0:
        return x0
    else:
        x0_val = np.sign(x0_val)

    # Find suitable interval
    left_points = - np.logspace(np.log10(-x0), np.log10(-a), 10, endpoint=True)
    right_points =  - np.logspace(-12, np.log10(0.001), 10)[::-1]
    curr = 0
    a0, b0 = left_points[curr], right_points[curr]
    target_a0, target_b0 = np.sign(target_fun(a0)), np.sign(target_fun(b0))
    if target_a0 == target_b0:
        flag = True
    else:
        flag = False
    while flag & (curr < 10):
        a1, b1 = left_points[curr], right_points[curr]
        target_a1, target_b1 = np.sign(target_fun(a1)), np.sign(target_fun(b1))
        if target_a1 != target_b0:
            flag = False
            a, b = a1, b0
        elif target_a0 != target_b1:
            flag = False
            a, b = a0, b1
        elif target_a1 != target_b1:
            flag = False
            a, b = a1, b1
        else:
            curr += 1
            a0, b0 = a1, b1
            target_a0, target_b0 = target_a1, target_b1

    # Check if continue
    if flag:
        raise ValueError('Could not find a suitable interval - the sign seems to be constant!')

    # Core of the computation
    zero = brentq(target_fun, a, b)
    obj_val = target_fun(zero)
    n_it = 0
    while (np.abs(obj_val) > tol) & (n_it < max_it):
        if obj_val > 0:
            zero = brentq(target_fun, a, zero)
        else:
            zero = brentq(target_fun, zero, b)
        
        obj_val = target_fun(zero)
        if obj_val > 0:
            b = zero
        else:
            a = zero
        
        n_it += 1
    
    if np.abs(obj_val) > tol:
        raise ValueError('Could not reach the desired tolerance!\nTry to increase max_it or reduce tol!')
    else:
        return zero

class Ch_RealizedRisk():
    '''
    Filtering approach proposed in [1]. It is based on the assumption of iid subordinated returns following the t-distribution.
    The characteristic function approach is used to recover the low-frequency risk measures.

    [1]: Gatta, F., & Lillo, F., & Mazzarisi, P. (2024). A High-frequency Approach to Risk Measures. TBD.

    Parameters:
    ----------------
        - theta: float
            the target probability level.
        - n_points: int, optional
            Number of points used to evaluate the ES (which is computed as the average of equi-spaced quantile estimates at probability levels theta_j <= theta). Default is 8.

    Example of usage
    ----------------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from utils import price2params
        from models import Ch_RealizedRisk

        df = pd.read_csv('stock.csv')
        df.index = pd.to_datetime(df.index)
        price = df.Close

        fitted_pars = price2params(price, c=78, sub_type='tpv') #Fit the t-distribution parameters

        mdl = Ch_RealizedRisk(0.05) #Initialize the model
        qf, ef = mdl.fit(78, fitted_pars, jobs=4) #Run to obtain the filtered VaR and ES
    
    Methods:
    ----------------
    '''
    def __init__(self, theta, n_points=8):
        self.theta = theta
        self.points = np.linspace(0, theta, n_points+1, endpoint=True)[1:]

    def quantile_from_cf_wrapper(self, theta, nu, mu, scale, pipend):
        '''
        Compute the daily risk measures from the high-frequency characteristic function, by using the Gil-Pelaez formula.
        The intra-day returns are assumed to be iid, t-distributed with 0 mean. The routine is used in the multiprocessing framework.

        INPUTS:
            - theta: float
                the target probability level.
            - nu: float
                the degrees of freedom.
            - mu: float
                the location parameter of the t-distribution.
            - scale: float
                the scale parameter of the t-distribution.
            - pipend: multiprocessing.Pipe
                The pipe to communicate the result to the main process.
        '''
        from scipy.integrate import quad #Load the quadrature function
        from scipy.optimize import minimize #Load function for minimization
        from scipy.special import kv, gamma #Load special functions for t-distribution pdf
        
        # Integrand of the Gil-Pelaez formula for the Student's t-distribution
        def gil_pelaez_integrand(t, x, nu, mu, scale, nu_2, normalization_term):
            adj_t = np.sqrt(nu) * scale * t
            cf_value = (kv(nu_2, adj_t) * np.exp( nu_2*np.log(adj_t) ) / normalization_term)**self.exp
            if mu == 0:
                return - np.sin(t * x) * cf_value / t
            else:
                return np.sin((self.exp*mu-x) * t) * cf_value / t

        # Objective function for the zero searching: CDF - theta
        def objective(x):
            integral, _ = quad(gil_pelaez_integrand, 0, np.inf, args=(x, nu, mu, scale, nu_2, normalization_term))
            return 0.5 - integral / np.pi - theta
        
        nu_2 = nu / 2
        normalization_term = 2**(nu_2 - 1) * gamma(nu_2)

        try:
            pipend.send(_custom_brentq(objective, -0.2, -0.001))
        except:
            temp_out = minimize(lambda x: np.abs(objective(x)), -0.001, bounds=[[-1, 0]], method='SLSQP')
            if temp_out.success and (temp_out.fun > 1e-4):
                temp_out = minimize(lambda x: np.abs(objective(x)), -0.001, bounds=[[-1, 0]], method='SLSQP', tol=1e-8)
                if temp_out.success and (temp_out.fun > 1e-4):
                    temp_out = minimize(lambda x: np.abs(objective(x)), -0.001, bounds=[[-1, 0]], method='SLSQP', tol=1e-12)
                    if temp_out.success and (temp_out.fun > 1e-4):
                        temp_out = minimize(lambda x: np.abs(objective(x)), -0.001, bounds=[[-1, 0]], method='SLSQP', tol=1e-16)
            
            if temp_out.fun > 1e-4:
                temp_out.success = False
            if not temp_out.success:
                temp_out.x = [np.nan]
            pipend.send(temp_out.x[0])
    
    def fit(self, c, internal_state, jobs=1):
        '''
        Map the t-distribution parameters into the low-frequency risk measures.

        INPUTS:
            - c: int
                the number of time indexes to sample.
            - internal_state: dict
                the fitted t-distribution parameters. Every key correspond to a day.
                Every value is assumed to be a list [degrees of freedom, location, scale].
            - jobs: int, optional
                Number of simultaneous processes to run. Default is 1.

        OUTPUTS:
            - qf: dict
                the filtered VaR. It has the same keys as internal_state.
            - ef: dict
                the filtered ES. It has the same keys as internal_state.
        '''
        import multiprocessing as mp #Load the multiprocessing module

        self.exp = c #Set the exponent for the characteristic function

        qf, ef = dict(), dict() #Initialize the dictionaries to store the daily values
        for day in internal_state.keys(): #Iterate over the days
            fitting_out = internal_state[day] #Get the fitting output for the day
            qf_list = list() # Initialize the list of quantile forecasts at different levels theta_j

            # Compute quantile in the inner theta_j
            for q_start in range(0, len(self.points), jobs): #Iterate over the theta_j
                # Create and start worker processes
                workers = list() # Initialize the list of workers
                end_point = np.min([q_start+jobs, len(self.points)]) # Define the end point of the iteration
                
                for theta_j in self.points[q_start:end_point]: # Iterate over theta_j
                    parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
                    worker = mp.Process(
                        target=self.quantile_from_cf_wrapper,
                        args=(theta_j, fitting_out[0], fitting_out[1],
                              fitting_out[2], child_pipend)) # Define the worker
                    workers.append([worker, parent_pipend]) # Append the worker to the list
                    worker.start() # Start the worker

                # Gather results from workers
                for worker, parent_pipend in workers:
                    q_emp = parent_pipend.recv() # Get the result from the worker
                    worker.join() # Wait for the worker to finish
                    qf_list.append(q_emp)

            qf[day] = qf_list[-1] #The VaR forecast is the last element of the list
            ef[day] = np.nanmean(qf_list) #The ES forecast is the average of the elements in the list

        return qf, ef

class Ch_RealizedRisk_MA():
    '''
    Filtering approach proposed in [1]. It is based on the assumption of MA(1) subordinated returns with t-distributed innovations.
    The characteristic function approach is used to recover the low-frequency risk measures.

    [1]: Gatta, F., & Lillo, F., & Mazzarisi, P. (2024). A High-frequency Approach to Risk Measures. TBD.

    Parameters:
    ----------------
        - theta: float
            the target probability level.
        - n_points: int, optional
            Number of points used to evaluate the ES (which is computed as the average of equi-spaced quantile estimates at probability levels theta_j <= theta). Default is 8.

    Example of usage
    ----------------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from utils import price2params_ma
        from models import Ch_RealizedRisk_MA

        df = pd.read_csv('stock.csv')
        df.index = pd.to_datetime(df.index)
        price = df.Close

        fitted_pars = price2params_ma(price, c=78, sub_type='tpv') #Fit the t-distribution parameters

        mdl = Ch_RealizedRisk_MA(0.05) #Initialize the model
        qf, ef = mdl.fit(78, fitted_pars, jobs=4) #Run to obtain the filtered VaR and ES
    
    Methods:
    ----------------
    '''
    def __init__(self, theta, n_points=8):
        self.theta = theta
        self.points = np.linspace(0, theta, n_points+1, endpoint=True)[1:]

    def quantile_from_cf_wrapper(self, theta, phi, nu, mu, scale, pipend):
        '''
        Compute the daily risk measures from the high-frequency characteristic function, by using the Gil-Pelaez formula.
        The intra-day returns are assumed to be iid, t-distributed with 0 mean. The routine is used in the multiprocessing framework.

        INPUTS:
            - theta: float
                the target probability level.
            - phi: float
                the autoregressive parameter.
            - nu: float
                the degrees of freedom.
            - mu: float
                the location parameter of the t-distribution.
            - scale: float
                the scale parameter of the t-distribution.
            - pipend: multiprocessing.Pipe
                The pipe to communicate the result to the main process.
        '''
        from scipy.integrate import quad #Load the quadrature function
        from scipy.optimize import minimize #Load function for minimization
        from scipy.special import kv, gamma #Load special functions for t-distribution pdf

        # Integrand of the Gil-Pelaez formula for the Student's t-distribution
        def gil_pelaez_integrand(t, x, phi, nu, mu, scale, nu_2, normalization_term):
            adj_t = np.sqrt(nu) * scale * t * (1+phi)
            cf_value = (kv(nu_2, adj_t) * np.exp( nu_2*np.log(adj_t) ) / normalization_term)**self.exp
            adj_t = np.sqrt(nu) * scale * t
            cf_value *= (kv(nu_2, adj_t) * np.exp( nu_2*np.log(adj_t) ) / normalization_term)
            adj_t = np.sqrt(nu) * scale * t * phi
            cf_value *= (kv(nu_2, adj_t) * np.exp( nu_2*np.log(adj_t) ) / normalization_term)
            if mu == 0:
                return - np.sin(t * x) * cf_value / t
            else:
                return np.sin(t * (self.exp*mu*(1+phi)-x)) * cf_value / t

        # Objective function for the zero searching: CDF - theta
        def objective(x):
            integral, _ = quad(gil_pelaez_integrand, 0, np.inf, args=(x, phi, nu, mu, scale, nu_2, normalization_term))
            return 0.5 - integral / np.pi - theta
        
        nu_2 = nu / 2
        normalization_term = 2**(nu_2 - 1) * gamma(nu_2)

        try:
            pipend.send(_custom_brentq(objective, -0.2, -0.001))
        except:
            temp_out = minimize(lambda x: np.abs(objective(x)), -0.001, bounds=[[-1, 0]], method='SLSQP')
            if temp_out.success and (temp_out.fun > 1e-4):
                temp_out = minimize(lambda x: np.abs(objective(x)), -0.001, bounds=[[-1, 0]], method='SLSQP', tol=1e-8)
                if temp_out.success and (temp_out.fun > 1e-4):
                    temp_out = minimize(lambda x: np.abs(objective(x)), -0.001, bounds=[[-1, 0]], method='SLSQP', tol=1e-12)
                    if temp_out.success and (temp_out.fun > 1e-4):
                        temp_out = minimize(lambda x: np.abs(objective(x)), -0.001, bounds=[[-1, 0]], method='SLSQP', tol=1e-16)
            
            if temp_out.fun > 1e-4:
                temp_out.success = False
            if not temp_out.success:
                temp_out.x = [np.nan]
            pipend.send(temp_out.x[0])
    
    def fit(self, c, internal_state, jobs=1):
        '''
        Map the t-distribution parameters into the low-frequency risk measures.

        INPUTS:
            - c: int
                the number of time indexes to sample.
            - internal_state: dict
                the fitted t-distribution parameters. Every key correspond to a day.
                Every value is assumed to be a list [autoregressive parameter, degrees of freedom, location, scale].
            - jobs: int, optional
                Number of simultaneous processes to run. Default is 1.

        OUTPUTS:
            - qf: dict
                the filtered VaR. It has the same keys as internal_state.
            - ef: dict
                the filtered ES. It has the same keys as internal_state.
        '''
        import multiprocessing as mp #Load the multiprocessing module

        self.exp = c-1 #Set the exponent for the characteristic function

        qf, ef = dict(), dict() #Initialize the dictionaries to store the daily values
        for day in internal_state.keys(): #Iterate over the days
            fitting_out = internal_state[day] #Get the fitting output for the day
            qf_list = list() # Initialize the list of quantile forecasts at different levels theta_j

            # Compute quantile in the inner theta_j
            for q_start in range(0, len(self.points), jobs):
                # Create and start worker processes
                workers = list() # Initialize the list of workers
                end_point = np.min([q_start+jobs, len(self.points)]) # Define the end point of the iteration
                
                for theta_j in self.points[q_start:end_point]: # Iterate over theta_j
                    parent_pipend, child_pipend = mp.Pipe() # Create a pipe to communicate with the worker
                    worker = mp.Process(
                        target=self.quantile_from_cf_wrapper,
                        args=(theta_j, fitting_out[0], fitting_out[1],
                              fitting_out[2], fitting_out[3], child_pipend)) # Define the worker
                    workers.append([worker, parent_pipend]) # Append the worker to the list
                    worker.start() # Start the worker

                # Gather results from workers
                for worker, parent_pipend in workers:
                    q_emp = parent_pipend.recv() # Get the result from the worker
                    worker.join() # Wait for the worker to finish
                    qf_list.append(q_emp)

            qf[day] = qf_list[-1]
            ef[day] = np.nanmean(qf_list)

        return qf, ef

class MC_RealizedRisk():
    '''
    Filtering approach proposed in [1]. It is based on the assumption of iid subordinated returns following the t-distribution.
    The Monte-Carlo approach is used to recover the low-frequency risk measures.

    [1]: Gatta, F., & Lillo, F., & Mazzarisi, P. (2024). A High-frequency Approach to Risk Measures. TBD.

    Parameters:
    ----------------
        - theta: float or list
            the target probability level, or a list of target probability levels.

    Example of usage
    ----------------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from utils import price2params
        from models import MC_RealizedRisk

        df = pd.read_csv('stock.csv')
        df.index = pd.to_datetime(df.index)
        price = df.Close

        fitted_pars = price2params(price, c=78, sub_type='tpv') #Fit the t-distribution parameters

        mdl = MC_RealizedRisk([0.05, 0.025]) #Initialize the model
        qf, ef = mdl.fit(78, fitted_pars) #Run to obtain the filtered VaR and ES
    
    Methods:
    ----------------
    '''
    def __init__(self, theta):
        self.theta = theta
    
    def simulate_iid(self, N, c, nu, mu, sigma, ant_v=True, seed=None):
        '''
        Monte-Carlo simulation for extracting the distribution of the low-frequency return.

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
    
    def fit(self, N, c, internal_state, ant_v=True, seed=None):
        '''
        Map the t-distribution parameters into the low-frequency risk measures.

        INPUTS:
            - N: int
                the number of Monte-Carlo paths to simulate (if ant_v==True, the number is doubled).
            - c: int
                the number of time indexes to sample.
            - internal_state: dict
                the fitted t-distribution parameters. Every key correspond to a day.
                Every value is assumed to be a list [degrees of freedom, location, scale].
            - ant_v: bool, optional
                Flag to indicate if the antithetic variates should be used. Default is True.
            - seed: int, optional
                Seed for the random number generator. Default is None.

        OUTPUTS:
            - qf: dict
                the filtered VaR. If self.theta is a float, it has the same keys as internal_state.
                If self.theta is a list, it has a key for every element in the list.
                Every value is a dict itself with the same keys as internal_state.
            - ef: dict
                the filtered ES. If self.theta is a float, it has the same keys as internal_state.
                If self.theta is a list, it has a key for every element in the list.
                Every value is a dict itself with the same keys as internal_state.
        '''
        # Initialize the dictionaries to store the daily values
        qf, ef = dict(), dict()
        if isinstance(self.theta, list):
            for theta in self.theta:
                qf[theta], ef[theta] = dict(), dict()

        for temp_key in internal_state.keys(): #Iterate over the days
            # Simulate the low-frequency returns
            temp_coeff = internal_state[temp_key]
            sim_data = self.simulate_iid(
                N, c, *temp_coeff, ant_v=ant_v, seed=seed)

            if isinstance(self.theta, list):
                for theta in self.theta: #Iterate over the target probability levels
                    q_val_temp = np.quantile(sim_data, theta) #Compute the VaR
                    qf[theta][temp_key] = q_val_temp
                    ef[theta][temp_key] =\
                        sim_data[ sim_data <= q_val_temp ].mean() #Compute the ES
            else:
                q_val_temp = np.quantile(sim_data, self.theta) #Compute the VaR
                qf[temp_key] = q_val_temp
                ef[temp_key] =\
                    sim_data[ sim_data <= q_val_temp ].mean() #Compute the ES

        return qf, ef

class MC_RealizedRisk_MA():
    '''
    Filtering approach proposed in [1]. It is based on the assumption of MA(1) subordinated returns with t-distributed innovations.
    The Monte-Carlo approach is used to recover the low-frequency risk measures.

    [1]: Gatta, F., & Lillo, F., & Mazzarisi, P. (2024). A High-frequency Approach to Risk Measures. TBD.

    Parameters:
    ----------------
        - theta: float or list
            the target probability level, or a list of target probability levels.

    Example of usage
    ----------------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from utils import price2params_ma
        from models import MC_RealizedRisk_MA

        df = pd.read_csv('stock.csv')
        df.index = pd.to_datetime(df.index)
        price = df.Close

        fitted_pars = price2params_ma(price, c=78, sub_type='tpv') #Fit the t-distribution parameters

        mdl = MC_RealizedRisk_MA([0.05, 0.025]) #Initialize the model
        qf, ef = mdl.fit(78, fitted_pars) #Run to obtain the filtered VaR and ES
    
    Methods:
    ----------------
    '''
    def __init__(self, theta):
        self.theta = theta
    
    def simulate_ma(self, N, c, phi, nu, mu, sigma, ant_v=True, seed=None):
        '''
        Monte-Carlo simulation for extracting the distribution of the low-frequency return.

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
    
    def fit(self, N, c, internal_state, ant_v=True, seed=None):
        '''
        Map the t-distribution parameters into the low-frequency risk measures.

        INPUTS:
            - N: int
                the number of Monte-Carlo paths to simulate (if ant_v==True, the number is doubled).
            - c: int
                the number of time indexes to sample.
            - internal_state: dict
                the fitted t-distribution parameters. Every key correspond to a day.
                Every value is assumed to be a list [degrees of freedom, location, scale].
            - ant_v: bool, optional
                Flag to indicate if the antithetic variates should be used. Default is True.
            - seed: int, optional
                Seed for the random number generator. Default is None.

        OUTPUTS:
            - qf: dict
                the filtered VaR. If self.theta is a float, it has the same keys as internal_state.
                If self.theta is a list, it has a key for every element in the list.
                Every value is a dict itself with the same keys as internal_state.
            - ef: dict
                the filtered ES. If self.theta is a float, it has the same keys as internal_state.
                If self.theta is a list, it has a key for every element in the list.
                Every value is a dict itself with the same keys as internal_state.
        '''
        # Initialize the dictionaries to store the daily values
        qf, ef = dict(), dict()
        if isinstance(self.theta, list):
            for theta in self.theta:
                qf[theta], ef[theta] = dict(), dict()

        for temp_key in internal_state.keys(): #Iterate over the days
            # Simulate the low-frequency returns
            temp_coeff = internal_state[temp_key]
            sim_data = self.simulate_ma(
                N, c, *temp_coeff, ant_v=ant_v, seed=seed)

            if isinstance(self.theta, list):
                for theta in self.theta: #Iterate over the target probability levels
                    q_val_temp = np.quantile(sim_data, theta) #Compute the VaR
                    qf[theta][temp_key] = q_val_temp
                    ef[theta][temp_key] =\
                        sim_data[ sim_data <= q_val_temp ].mean() #Compute the ES
            else:
                q_val_temp = np.quantile(sim_data, self.theta) #Compute the VaR
                qf[temp_key] = q_val_temp
                ef[temp_key] =\
                    sim_data[ sim_data <= q_val_temp ].mean() #Compute the ES

        return qf, ef
