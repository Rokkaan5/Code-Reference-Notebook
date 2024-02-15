# %%
"""
title: ASTR 5550: HW helper functions
author: Jasmine Kobayashi

Functions that would (likely) be helpful to save and be able to 
reuse throughout all assignments, etc. for ASTR 5550
"""
# %%
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



# %%
class helper:
    def __init__(self,x_min=0,x_max=15,step=1):
        self.step = step
        self.x = np.arange(x_min,x_max+self.step,self.step)

    def Expectation(self,fx,Px):
        E = np.sum(fx*Px)              # E(x) = sum(f(x)*P(x))
        return E
    
    def calculate_mean_mode_sigma(self,x,Px):
        # calculate mean: mu = E(x)
        self.mu = self.Expectation(fx=x,Px=Px)   

        # calculate stand. dev: sigma = sqrt(sigma^2); sigma^2 = E(x-mu)^2
        self.sigma = np.sqrt(self.Expectation(fx=(x-self.mu)**2,Px=Px)) 

        # calculate mode: x value where P(x) is highest
        self.mode = x[np.argmax(Px)]

    def plot_mark_mean(self,
                       Px : np.array,
                       ymin=0,
                       ymax=None,
                       colors= 'purple',
                       linestyles='dashdot',
                       label = "Mean: "):
        if ymax is None:
            ymax = max(Px)
        
        # mark mean in plot
        plt.vlines(self.mu, ymin=ymin, ymax=ymax,
                   linestyles=linestyles,
                   colors=colors,
                   label=label + "$\mu$ = {0:.3f}".format(self.mu))
        plt.legend()

    def plot_mark_mode(self,
                       Px : np.array,
                       ymin=0,
                       ymax=None,
                       colors='black',
                       linestyles='solid',
                       label='Most Likely Value: '):
        if ymax is None:
            ymax = max(Px)

        # mark mode in plot
        plt.vlines(self.mode,ymin=ymin,ymax=ymax,
                   linestyles=linestyles,
                   colors=colors,
                   label=label+ "mode = {0:.3f}".format(self.mode))
        plt.legend()

    def plot_mark_std(self,
                      Px : np.array,
                       ymin=0,
                       ymax=None,
                      colors='orange',
                      linestyles='dotted',
                      label="Standard deviation: "):
        if ymax is None:
            ymax = max(Px)

        # mark standard deviation (wrt mean) in plot
        plt.vlines(self.mu+self.sigma,
                   ymin=ymin,ymax=ymax,
                   colors=colors,
                   linestyles=linestyles,
                   label=label + "$\sigma$ = {0:.3f}".format(self.sigma))
        plt.vlines(self.mu-self.sigma,
                   ymin=ymin,ymax=ymax,
                   colors=colors,
                   linestyles=linestyles)
        plt.legend()

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
# &E(x) = \mu \\
# &E(x-\mu)^2 = \sigma^2
# \end{align*}
# 
# where, $E(f(x)) = \sum f(x)P(x)$

# %% Binomial Probability
class binomial(helper):
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
        self.q = 1-self.p

    def P_binomial(self, x, n, p, q):                             
        """Function to calculate Binomial Probability P(x;n,p)

        Pb(x;n,p) = {(n!)/[(n-x)!(x!)]} * p^x * q^(n-x)
        """
        # formula components
        numerator = factorial(n)                                # fraction numerator: n!
        denominator = factorial(n-x)*factorial(x)               # fraction denominator: (n-x)!x!
        fraction = numerator/denominator
        px = p**x                                               # p^x
        qnx = q**(n-x)                                          # q^(n-x) = (1-p)^(n-x)
        
        # Binomial probability
        Pb = fraction*px*qnx                                    # [n!/((n-x)!x!)]*(p^x)*(q*(n-x))
        return Pb
    
    def plot_binomial(self,title="Binomial Distribution",label="Binomial",color='tab:blue'):
        self.Px = self.P_binomial(x= self.x,n=self.n,p=self.p,q=self.q)
        
        plt.plot(self.x,self.Px, label=label,c=color)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("P(x)")

    def complete_ind_binomial_plot(self,title="Binomial Distribution"):
        self.plot_binomial(title=title,label=None)
        self.calculate_mean_mode_sigma(x=self.x,Px=self.Px)
        self.plot_mark_mean(Px=self.Px)
        self.plot_mark_mode(Px=self.Px)
        self.plot_mark_std(Px=self.Px)

# %% [markdown]
# # Poisson probability
# 
# \begin{align*}
# P_P(x;\mu) &= \frac{\mu^x}{x!}e^{-\mu}
# \end{align*}

# %% Poisson
class poisson(helper):
    def __init__(self, x_min=0, x_max=15, step=1,mu=3.5):
        super().__init__(x_min, x_max, step)              
        self.mu = mu                          # the term "lambda" has its own purpose in python so I named the variable "lmd" instead

    # function to calculate poisson probability P(x;lambda)
    def P_poisson(self,x : int, mu : float): 
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
    
    def plot_poisson(self,title="Poisson Distribution",label="Poisson",color='tab:orang'):
        self.Px = self.P_poisson(x=self.x,mu=self.mu)
        
        plt.plot(self.x,self.Px, label=label,c=color)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("P(x)")

    def complete_ind_poisson_plot(self,title="Poisson Distribution"):
        self.plot_poisson(title=title,label=None)
        self.calculate_mean_mode_sigma(x=self.x,Px=self.Px)
        self.plot_mark_mean(Px=self.Px)
        self.plot_mark_mode(Px=self.Px)
        self.plot_mark_std(Px=self.Px)

# %% [markdown]
# # Gaussian probability
# 
# \begin{align*}
# P_G(x;\mu, \sigma) &= \frac{1}{\sigma \sqrt{2 \pi}}\exp\left({\frac{-(x-\mu)^2}{2\sigma^2}}\right)
# \end{align*}

# %%
class gaussian(helper):
    def __init__(self, mu, sigma, x_min=0, x_max=15, step=1):
        super().__init__(x_min, x_max, step)
        
        self.mu = mu
        self.sigma = sigma

    def P_gaussian(self,x,mu,sigma:float):
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
    
    def plot_gaussian(self,title = 'Gaussian Distribution',label='Gaussian',color = "tab:green"):
        self.Px = self.P_gaussian(x=self.x,mu=self.mu,sigma=self.sigma)

        plt.plot(self.x,self.Px, label=label,c=color)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("P(x)")

# %% [markdown]
# # Posterior probability (Bayesian theory)
# 
# \begin{align*}
# P(t;x) = \frac{P(x;t)P(t)}{P(x)}
# \end{align*}

# ## Volcano age example
# **The following code is more specific to the problem from HW2, which was as follows:**
#
# The age of a volcanic eruption on Mars can be inferred from counting the surface density of craters of a given diameter or greater. It is established that the impact rate $r=0.01$ craters $\text{km}^{-2}\text{Myr}^{-1}$. You survey an area ($A$) of 10 $\text{km}^2$ and find 3 craters ($x$).

# One can explore this problem further by calculating the *posterior* probability, $P(x;t)$ over a range of times (say, $t=0$ to 150 in Myr intervals) assuming a Poisson parent distribution and knowing $x=3$. Do this problem on your computer. 

# Important point: the initial calculation yields a set of *relative* probabilities; under Bayes' law: 
# $$P(t;x) = \frac{P(x;t)P(t)}{P(x)}$$

# Since we have no prior knowledge in this case, we treat $P(t)/P(x)$ as a constant; You must normalize $P(t;x)$ so that the total probability is $1$. $P(t;x)$ will have units of "probability/Myr". Plot your results. Mark the mean and mode (most likely age) and directly compute the standard deviation. Compare these values with your simple estimate.
# %%
class Posterior_Probability(poisson):
    def __init__(self,t0=0,tf=150,step=1,obs_area=10):
        super().__init__(step)
        self.t = np.arange(t0,tf+step,step)
        self.obs_area = obs_area

    def post_prob(self,x):
        # P(x;t)/A => probability/km^2
        self.Pp = self.P_poisson(x=x,mu=self.t.astype('float'))/self.obs_area

        #constant P(t)/P(x) = 1/sum(P(x;t))
        self.Pt_Px = 1/np.sum(self.Pp)

        #posterior probability
        self.posterior = self.Pp*self.Pt_Px
        return self.posterior


# %% [markdown]
# # Random Walk
# %%
class rand_walk(binomial,gaussian):
    def __init__(self, step=1, n=100, p=0.5):
        super().__init__(step, n, p)
        self.x = np.arange(0,self.n+step,step)
        self.d = self.distance(x=self.x,n=self.n)

        self.P_right = p
    
    def distance(self,x,n):     
        d = 2*x -n
        return d
    
    def get_xn(self,ntrial,normalize = True):
        self.xn = np.zeros(len(self.d))
        self.wh = np.where(self.d == 0.0)
        for _ in range(ntrial):
            i = self.wh[0]
            for _ in range(int(self.n/2)):
                if (rnd.random() <= self.P_right):
                    i = i+self.step
                else:
                    i = i-self.step
            self.xn[int(np.round(i))] += 1.0
        
        if normalize:
            self.xn = self.xn/ntrial
        
        return self.xn
 
    def get_mu_sig(self):
        self.mu = np.sum(self.x)/self.n
        self.sigma = np.sqrt(self.n)
        return

    def plot_rand_walk_BG(self,title='Random Walk (Binomial & Gaussian)'):
        self.get_mu_sig()
        self.Pb = self.P_binomial(x=self.x,n=self.n,p=self.p,q=self.q) 
        self.Pg = self.P_gaussian(x=self.x,mu=self.mu,sigma=self.sigma)

        plt.figure()
        plt.plot(self.d,self.Pb,label = 'Binomial')
        plt.plot(self.d,self.Pg,'--', label = 'Gaussian')
        plt.xlabel('d')
        plt.ylabel('P(d)')
        plt.title(title)
        plt.legend()
        plt.show()

# %% [markdown]
# # Random walk with Slope
# What if, however, the lamppost ($x=0$) is at the bottom of a parabolic valley as pictured below. 
# Let the altitude be $z=0.005 d^2$, so that the $\text{slope} = 0.01 d$. 
# Now suppose the drunk tends to stagger downhill. Let the probability of a right step be $0.5-\text{slope}$. 
# With a brute force algorithm, calculate $P(d)$. Overplot your results. What is $\sigma$?

# ![valley](CU-Boulder\AstroPhys\HW\valley.PNG)

# Hint: Implement a random walk program with $n=100$ steps by calling a uniform random number 
# then comparing to the right-step probability, which is a function of position. 
# Record the end position. Loop ~100,000 times and make a histogram of the end positions.

# %%
class rand_slope_walk(rand_walk):
    def __init__(self, slope_const=0.01, n=100, step=1, p=0.5):
        super().__init__(step, n, p)

        self.slope = slope_const*self.d             # defaults to slope = 0.01d
        self.P_right = p - self.slope

    def get_xn(self,ntrial,normalize = True):
        self.xn = np.zeros(len(self.d))
        self.wh = np.where(self.d == 0.0)
        for _ in range(ntrial):
            i = self.wh[0]
            for _ in range(int(self.n/2)):
                if (rnd.random() <= self.P_right[i]):
                    i = i+self.step
                else:
                    i = i-self.step
            self.xn[int(np.round(i))] += 1.0
        
        if normalize:
            self.xn = self.xn/ntrial
        
        return self.xn
    
    def get_mu_sig(self):
        self.mu = np.sum(self.d*self.xn)
        self.xbar2 = np.sum((self.d.astype('float')**2)*self.xn)
        self.sigma = np.sqrt(self.xbar2 - self.mu)
        return 
    
    def plot_walk_count(self,title='Random Walk Distribution',
                        ntrial=1000,
                        bar_width=15.0,
                        normalize_xn=True):
        self.get_xn(ntrial,normalize=normalize_xn)

        plt.bar(self.d,self.xn,width=bar_width)
        plt.xlabel('d')
        plt.ylabel('count')
        plt.title(title)
        plt.show()

# %% [markdown]
# # Not so Random Walk: Power Law
# 
# In this problem, we repeat the previous problem with a new caveat. His/her step size is 0.50 (whatever units) but increases with absolute value of his/her current position: 
# $$\text{Step} = 0.5 + |x| * 0.025$$
# 
# Let the number of steps be $n=400$ (half the original step size, so four times the number of steps). Implement a random walk program with $n=400$ steps by calling a uniform random number then comparing to the right-step probability, in this case 1/2. Record the end position. Loop ~100,000 times and make a histogram of the end positions.
# 
# Calculate $\sigma$. The mean should be zero. Overplot a Gaussian with the calculated $\sigma$. Is the resulting distribution a Gaussian?

# %%
class rand_power_law(rand_walk):
    def __init__(self, n=100, p=0.5, step0_size=0.5):
        super().__init__(n, p)
        self.P_right = p
        self.step0_size = step0_size

    def nearest_value_idx(self,x,value):
        idx = np.abs(np.asarray(x) - value).argmin()
        return idx

    def get_xn(self,ntrial,normalize = True):
        self.xn = np.zeros(len(self.d))
        self.wh = np.where(self.d == 0.0)
        for _ in range(ntrial):
            i = self.wh[0]
            step = self.step0_size
            for _ in range(int(self.n/2)):
                
                if (rnd.random() <= self.P_right):
                    i = i+step
                else:
                    i = i-step
                nearest_x = self.nearest_value_idx(self.x,i)
                step = 0.5 + np.abs(self.x[nearest_x])*0.025
            self.xn[nearest_x] += 1.0
        
        if normalize:
            self.xn = self.xn/ntrial
        
        return self.xn
    


