"""
title: ASTR 5550: HW helper functions
author: Jasmine Kobayashi

Functions that would (likely) be helpful to save and be able to 
reuse throughout all assignments, etc. for ASTR 5550
"""
# %% libraries
# libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from scipy.special import factorial
from scipy.signal import peak_widths
# %% Full-Width Half-Max
def get_fwhm(pdf:np.array, return_max_idx = False):
    """Simple function to get Full-Width Half-Max (FWHM)
    
    Relevant to one way to calculate standard deviation (sigma) 
    FWHM = sigma [8 ln(2)]^(1/2)

    Parameters
    ----------
    pdf : np.array
        probability distribution "function" to find FWHM from
    return_max_idx : boolean
        indicate whether or not to return the max value and index of max value in addition to FWHM value

    Returns
    -------
    fwhm : float
        fwhm value calculated from pdf
    """
    pdf_max = max(pdf)
    where_max = np.argmax(pdf)
    fwhm = peak_widths(pdf,where_max,rel_height=0.5)[0]
    
    if return_max_idx:
        return fwhm, pdf_max, where_max

    return fwhm



# %% General Helper Functions
class helper:
    # TODO: add documentation to these functions
    def __init__(self,x_min=0,x_max=15,step=1):
        self.step = step
        self.x = np.arange(x_min,x_max+self.step,self.step)
        self.Px = None
        self.mu = None
        self.sigma = None
        self.mode = None

    def Expectation(self,fx,Px):
        E = np.sum(fx*Px)              # E(x) = sum(f(x)*P(x))
        return E
    
    def expected_mean(self,x,Px):
        # calculate mean: mu = E(x)
        mu = self.Expectation(fx=x,Px=Px) 
        return mu  

    def expected_sigma(self,mu, x,Px):
        # calculate stand. dev: sigma = sqrt(sigma^2); sigma^2 = E(x-mu)^2
        sigma = np.sqrt(self.Expectation(fx=(x-mu)**2,Px=Px)) 
        return sigma

    def calculate_mode(self,x,Px):
        # calculate mode: x value where P(x) is highest
        mode = x[np.argmax(Px)]
        return mode
    
    def calculate_expected_mean_mode_sigma(self,x,Px,set_to_object=False):
        mu = self.expected_mean(x=x,Px=Px)
        sigma = self.expected_sigma(mu=mu,x=x,Px=Px)
        mode = self.calculate_mode(x=x,Px=Px)

        if set_to_object:
            self.mu = mu
            self.sigma = sigma
            self.mode = mode
            
            return self.mu,self.mode,self.sigma
        else:
            return mu,mode,sigma
        

    def plot_mark_mean(self,
                       mu,
                       ymin=0,
                       ymax=None,
                       colors= 'purple',
                       linestyles='dashdot',
                       label = "Mean: "):
        if ymax is None:
            ymax = max(self.Px)
        
        # mark mean in plot
        plt.vlines(mu, ymin=ymin, ymax=ymax,
                   linestyles=linestyles,
                   colors=colors,
                   label=label + "$\mu$ = {0:.3f}".format(mu))
        plt.legend()

    def plot_mark_mode(self,
                       mode,
                       ymin=0,
                       ymax=None,
                       colors='black',
                       linestyles='solid',
                       label='Most Likely Value: '):
        if ymax is None:
            ymax = max(self.Px)

        # mark mode in plot
        plt.vlines(mode,ymin=ymin,ymax=ymax,
                   linestyles=linestyles,
                   colors=colors,
                   label=label+ "mode = {0:.3f}".format(mode))
        plt.legend()

    def plot_mark_std(self,
                      sigma,
                      mu=None,
                       ymin=0,
                       ymax=None,
                      colors='orange',
                      linestyles='dotted',
                      label="Standard deviation: "):
        if ymax is None:
            ymax = max(self.Px)

        if mu is None:
            mu = 0
            print("mu was not passed so mu = 0 was used (i.e. +/- sigma is centered at x=0)")

        # mark standard deviation (wrt mean) in plot
        plt.vlines(mu+sigma,
                   ymin=ymin,ymax=ymax,
                   colors=colors,
                   linestyles=linestyles,
                   label=label + "$\sigma$ = {0:.3f}".format(sigma))
        plt.vlines(mu-sigma,
                   ymin=ymin,ymax=ymax,
                   colors=colors,
                   linestyles=linestyles)
        plt.legend()

    def comprehensive_plot(self,mark_mode = False):
        self.plot_mark_mean(mu=self.mu)
        self.plot_mark_std(sigma=self.sigma,mu=self.mu)
        if mark_mode:
            self.mode = self.calculate_mode(x=self.x,Px=self.Px)


# %% [markdown]
# # Binomial probability:
# 
# \begin{align*}
# P_B(x;n,p) &= \frac{n!}{(n-x)!x!}p^x q^{n-x}
# \end{align*}
# 
# where, $q=1-p$
# 
# And other useful:
# 
# \begin{align*}
# \mu &= \displaystyle \sum_{x=0}^n x \frac{n!}{(n-x)!x!}p^x q^{n-x} \\
# &= np \\
# \sigma^2 &= \displaystyle \sum_{x=0}^n (x- \mu )^2 \frac{n!}{(n-x)!x!}p^x q^{n-x} \\
# &= np(1-p)
# \end{align*}
# 


# %% Binomial Probability
        
def P_binomial(x, n, p):                             
        """Function to calculate Binomial Probability P(x;n,p)

        Pb(x;n,p) = {(n!)/[(n-x)!(x!)]} * p^x * q^(n-x)
        """
        q = 1-p
        # formula components
        numerator = factorial(n)                                # fraction numerator: n!
        denominator = factorial(n-x)*factorial(x)               # fraction denominator: (n-x)!x!
        fraction = numerator/denominator
        px = p**x                                               # p^x
        qnx = q**(n-x)                                          # q^(n-x) = (1-p)^(n-x)
        
        # Binomial probability
        Pb = fraction*px*qnx                                    # [n!/((n-x)!x!)]*(p^x)*(q*(n-x))
        return Pb

def binomial_mean_sigma(n, p):
        mean = n*p
        sigma = np.sqrt(n*p*(1-p))
        return mean, sigma

class binomial_distribution(helper):
    # TODO: add documentation to the rest of these functions
    def __init__(self, x_min=0, x_max=15, step=1,n=100,p=0.5):
        """Functions related to using Binomial probability

        Parameters
        ----------
        x_min : numeric
            minimum x value of x array
        x_max : numeric
            maximum x value of x array
        step : numeric
            step size between each element in x array
        n : numeric
            number of 'trials'
        p : float
            probability of 'success' in single trial
        """
        super().__init__(x_min, x_max, step)
        self.n = n
        self.p = p

        self.Px = P_binomial(x=self.x,n=self.n,p=self.p)
    
    def calculate_mean_mode_sigma(self):
        self.mu, self.sigma = binomial_mean_sigma(n=self.n,p=self.p)
        self.mode = self.calculate_mode(x=self.x,Px=self.Px)

    def plot_binomial(self,title="Binomial Distribution",
                      label="Binomial",color='tab:blue',
                      comprehensive=True,mark_mode=False):
        plt.plot(self.x,self.Px, label=label,c=color)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("P(x)")
        if comprehensive:
            self.calculate_mean_mode_sigma()
            self.comprehensive_plot(mark_mode=mark_mode)


# %% [markdown]
# # Poisson probability
# 
# \begin{align*}
# P_P(x;\mu) &= \frac{\mu^x}{x!}e^{-\mu}
# \end{align*}
#   
# And other useful:
# 
# \begin{align*}
# \sigma^2 &=  \displaystyle \sum_{x=0}^n \frac{\mu^x}{x!}e^{-\mu} \\
# &= \mu
# \end{align*}
# 

# %% Poisson
# function to calculate poisson probability P(x;lambda)
def P_poisson(x : int, mu : float): 
    """Function to calculate Poisson probability P(x;mu)
    
    Parameters
    ----------
    x : int 
        x value we want to get probability of
    mu : float
        mean occurence of value x

        This one especially HAS to be a float for this probability function,
        otherwise the `np.power(mu,x)` (to calculate mu^x portion) doesn't work properly and 
        will end up with weird (like negative) probabilities

    Returns
    -------
    Pp : float
        Poisson probability of x
    """
    # formula components                            
    numerator = np.power(mu,x)                  # fraction numerator: mu^x (this one can weird if mu are not floats)
    denominator = factorial(x)                  # fraction denominator: x!
    fraction = numerator/denominator
    e_mu = np.exp(-mu)                          # e^(-mu)
    
    # Poisson probability
    Pp = fraction*e_mu                          # P(x;mu) = [(mu^x)/x!] * e^(-mu)
    return Pp

def poisson_sigma(mu):
    sigma = np.sqrt(mu)
    return sigma

class poisson_distribution(helper):
    # TODO: add documentation to the rest of these functions
    def __init__(self, x_min=0, x_max=15, step=1,mu=3.5):
        super().__init__(x_min, x_max, step)              
        self.mu = mu    

        self.Px = P_poisson(x=self.x,mu=self.mu)     

    def calculate_mean_mode_sigma(self):
        self.sigma = poisson_sigma(mu=self.mu)
        self.mode = self.calculate_mode(x=self.x,Px=self.Px)
    
    def plot_poisson(self,title="Poisson Distribution",
                     label="Poisson",color='tab:orange',
                     comprehensive=True,mark_mode=False):
        self.Px = P_poisson(x=self.x,mu=self.mu)
        
        plt.plot(self.x,self.Px, label=label,c=color)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("P(x)")

        if comprehensive:
            self.calculate_mean_mode_sigma()
            self.comprehensive_plot(mark_mode=mark_mode)


# %% [markdown]
# # Gaussian probability
# 
# \begin{align*}
# P_G(x;\mu, \sigma) &= \frac{1}{\sigma \sqrt{2 \pi}}\exp\left({\frac{-(x-\mu)^2}{2\sigma^2}}\right)
# \end{align*}

# %% Gaussian
def P_gaussian(x,mu,sigma:float):
        """Function to calculate Poisson probability P(x;mu)
        
        Parameters
        ----------
        x : int 
            x value we want to get probability of
        mu : float
            mean occurence of value x
        sigma : float
            standard deviation of x

        Returns
        -------
        Pg : float
            Gaussian probability of x
        """
        # formula components
        pow_num = -((x-mu)**2)                  # numerator of exponential power
        pow_den = 2*(sigma**2)                  # denominator of exponential power
        power = pow_num/pow_den                 # power of exponential: -(x-<x>)^2/2(sigma^2)
        
        denominator = sigma*np.sqrt(2*np.pi)    # fraction denominator: sigma*squareroot(2*pi)
        
        # Gaussian probability
        Pg = (1/denominator)*np.exp(power)      # P(x;mu,sigma) 
        return Pg

class gaussian_distribution(helper):
    # TODO: add documentation to the rest of these functions
    def __init__(self, mu, sigma, x_min=0, x_max=15, step=1):
        super().__init__(x_min, x_max, step)
        
        self.mu = mu
        self.sigma = sigma

        self.Px = P_gaussian(x=self.x,mu=self.mu,sigma=self.sigma)
    
    def plot_gaussian(self,title = 'Gaussian Distribution',
                      label='Gaussian',color = "tab:green",
                      comprehensive=True,mark_mode=False):
        self.Px = P_gaussian(x=self.x,mu=self.mu,sigma=self.sigma)

        plt.plot(self.x,self.Px, label=label,c=color)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("P(x)")

        if comprehensive:
            self.mode = self.calculate_mode(x=self.x,Px=self.Px)
            self.comprehensive_plot(mark_mode=mark_mode)


