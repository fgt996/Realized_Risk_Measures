
import numpy as np

class PinballLoss():
    '''
    Pinball (a.k.a. Quantile) loss function

    Parameters:
    ----------------
        - theta: float
            the target confidence level
        - ret_mean: bool, optional
            if True, the function returns the mean of the loss, otherwise the loss point-by-point. Default is True

    Example of usage
    ----------------
    .. code-block:: python

        import numpy as np
        from eval_utils import PinballLoss

        y = np.random.randn(250)*1e-2  #Replace with price returns
        qf = np.random.uniform(-1, 0, 250)  #Replace with quantile forecasts
        theta = 0.05 #Set the desired confidence level

        PinballLoss(theta)(qf, y) #Compute the pinball loss
    
    Methods:
    ----------------
    '''
    def __init__(self, theta, ret_mean=True):
        self.theta = theta
        self.ret_mean = ret_mean
    
    def __call__(self, y_pred, y_true):
        '''
        Compute the pinball loss

        INPUTS:
            - y_pred: ndarray
                the predicted values
            - y_true: ndarray
                the true values

        OUTPUTS:
            - loss: float
                the loss function mean value, if ret_mean is True. Otherwise, the loss for each observation
        '''
        #Check consistency in the dimensions
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1,1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1,1)
        if y_pred.shape != y_true.shape:
            raise ValueError(f'Dimensions of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) do not match!!!')
        # Compute the pinball loss
        error = y_true - y_pred
        loss = np.where(error >= 0, self.theta * error, (self.theta - 1) * error)
        if self.ret_mean: #If true, return the mean of the loss
            loss = np.nanmean(loss)
        return loss

class barrera_loss():
    '''
    Barrera loss function
    '''
    def __init__(self, theta, ret_mean=True):
        '''
        INPUT:
        - theta: float,
            the threshold for the loss function
        - ret_mean: bool,
            if True, the function returns the mean of the loss. Default is True
        '''
        self.theta = theta
        self.ret_mean = ret_mean
    
    def __call__(self, v_, e_, y_):
        '''
        INPUT:
        - v_: numpy array,
            the quantile estimate
        - e_: numpy array,
            the expected shortfall estimate
        - y_: numpy array,
            the actual time series
        OUTPUT:
        - loss: float,
            the loss function mean value, if ret_mean is True. Otherwise, the loss for each observation
        '''
        v, e, y = v_.flatten(), e_.flatten(), y_.flatten()
        r = e - v #Barrera loss is computed on the difference ES - VaR
        if self.ret_mean: #If true, return the mean of the loss
            loss = np.nanmean( (r - np.where(y<v, (y-v)/self.theta, 0))**2 )
        else: #Otherwise, return the loss for each observation
            loss = (r - np.where(y<v, (y-v)/self.theta, 0))**2
        return loss

class patton_loss():
    '''
    Patton loss function
    '''
    def __init__(self, theta, ret_mean=True):
        '''
        INPUT:
        - theta: float,
            the threshold for the loss function
        - ret_mean: bool,
            if True, the function returns the mean of the loss. Default is True
        '''
        self.theta = theta
        self.ret_mean = ret_mean
    
    def __call__(self, v_, e_, y_):
        '''
        INPUT:
        - v_: numpy array,
            the quantile estimate
        - e_: numpy array,
            the expected shortfall estimate
        - y_: numpy array,
            the actual time series
        OUTPUT:
        - loss: float,
            the loss function mean value, if ret_mean is True. Otherwise, the loss for each observation
        '''
        v, e, y = v_.flatten()*100, e_.flatten()*100, y_.flatten()*100
        if self.ret_mean: #If true, return the mean of the loss
            loss = np.nanmean(
                np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e) - 1
            )
        else: #Otherwise, return the loss for each observation
            loss = np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e) - 1
        return loss

class bootstrap_mean_test():
    '''
    Bootstrap test for assessing whenever mean of a sample is == or >= a target value

    Parameters:
    ----------------
            - mu_target: float
                the mean to test against
            - one_side: bool, optional
                if True, the test is one sided (i.e. H0: mu >= mu_target), otherwise it is two-sided (i.e. H0: mu == mu_target). Default is False
            - n_boot: int, optional
                the number of bootstrap replications. Default is 10_000
    '''
    def __init__(self, mu_target, one_side=False, n_boot=10_000):
        self.mu_target = mu_target
        self.one_side = one_side
        self.n_boot = n_boot
    
    def null_statistic(self, B_data):
        '''
        Compute the null statistic for the bootstrap sample

        INPUTS:
            - B_data: ndarray
                the bootstrap sample

        OUTPUTS:
            - stat: float
                the null statistic
        
        :meta private:
        '''
        return (np.mean(B_data) - self.obs_mean) * np.sqrt(self.n) / np.std(B_data)
    
    def statistic(self, data):
        '''
        Compute the test statistic for the original sample

        INPUTS:
            :data: ndarray
            the original sample

        OUTPUTS:
            - :float
                the test statistic

        :meta private:
        '''
        return (self.obs_mean - self.mu_target) * np.sqrt(self.n) / np.std(data)
    
    def __call__(self, data, seed=None):
        '''
        Compute the test

        INPUTS:
            - data: ndarray
                the original sample
            - seed: int, optional
                the seed for the random number generator. Default is None

        OUTPUTS:
            - statistic: float
                the test statistic
            - p_value: float
                the p-value of the test
        '''
        np.random.seed(seed)

        self.obs_mean = np.mean(data)
        self.n = len(data)

        B_stats = list()
        for _ in range(self.n_boot):
            B_stats.append( self.null_statistic(
                np.random.choice(data, size=self.n, replace=True) ))
        B_stats = np.array(B_stats)
        self.B_stats = B_stats
        
        if self.one_side:
            obs = self.statistic(data)
            return {'statistic':obs, 'p_value':np.mean(B_stats < obs)}
        else:
            obs = np.abs(self.statistic(data))
            return {'statistic':self.statistic(data),
                    'p_value':np.mean((B_stats > obs) | (B_stats < -obs))}

class AS14_test(bootstrap_mean_test):
    '''
    Acerbi-Szekely test for assessing the goodness of the Expected Shortfall estimate, with both Z1 and Z2 statistics, as described in:

    Acerbi, C., & Szekely, B. (2014). Back-testing expected shortfall. Risk, 27(11), 76-81.

    Parameters:
    ----------------
            - one_side: bool, optional
                if True, the test is one sided (i.e. H0: mu >= mu_target). Default is False
            - n_boot: int, optional
                the number of bootstrap replications. Default is 10_000

    Example of usage
    ----------------
    .. code-block:: python

        import numpy as np
        from eval_utils import AS14_test

        y = np.random.randn(250)*1e-2  #Replace with price returns
        qf = np.random.uniform(-1, 0, 250)*1e-1  #Replace with quantile forecasts
        ef = np.random.uniform(-1, 0, 250)*1e-1  #Replace with expected shortfall forecasts
        theta = 0.05 #Set the desired confidence level

        # Compute the Acerbi-Szekely test with Z1 statistic
        AS14_test()(qf, ef, y, test_type='Z1', theta=theta, seed=2)
    
    Methods:
    ----------------
    '''
    def __init__(self, one_side=False, n_boot=10_000):
        super().__init__(-1, one_side, n_boot)
    
    def as14_transform(self, test_type, Q, E, Y, theta):
        '''
        Transform the data to compute the Acerbi-Szekely test

        INPUTS:
            :test_type: str
            the type of test to perform. It must be either 'Z1' or 'Z2'
            :Q: ndarray
            the quantile estimates
            :E: ndarray
            the expected shortfall estimates
            :Y: ndarray
            the actual time series
            :theta: float
            the threshold for the test

        OUTPUTS:
            - :ndarray
                the transformed data

        :meta private:
        '''
        import warnings

        Q, E, Y = Q.flatten(), E.flatten(), Y.flatten() #Flatten the data
        if test_type == 'Z1':
            output = (- Y/E)[Y <= Q]
        elif test_type == 'Z2':
            output = - Y * (Y <= Q) / (theta * E)
        else:
            raise ValueError(f'test_type {test_type} not recognized. It must be either Z1 or Z2')
        n = len(output)
        output = output[~np.isnan(output)]
        if len(output) < n:
            warnings.warn('There are NaN in the population! They have been removed.', UserWarning)
        return output

    def __call__(self, Q, E, Y, theta, test_type='Z1', seed=None):
        '''
        Compute the test

        INPUTS:
            - Q: ndarray
                the quantile estimates
            - E: ndarray
                the expected shortfall estimates
            - Y: ndarray
                the actual time series
            - test_type: str, optional
                the type of test to perform. It must be either 'Z1' or 'Z2'. Default is 'Z1'
            - seed: int, optional
                the seed for the random number generator. Default is None
            
        OUTPUTS:
            - statistic: float
                the test statistic
            - p_value: float
                the p-value of the test
        '''
        return super().__call__( self.as14_transform(test_type, Q, E, Y, theta).flatten(), seed)

