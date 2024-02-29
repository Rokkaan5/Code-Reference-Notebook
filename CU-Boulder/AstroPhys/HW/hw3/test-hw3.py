# Test-script for HW3
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # 1. Joint Probability and Find the Gap
# Suppose the goal is to take a series of images of galaxies which necessitates very long exposures in a narrow field-of-view (FOV). The expected (mean) number of galaxies in this FOV is $19.3$.

# %% [markdown]
# ## Part (a)
# How many exposures in separate FOVs are needed so that the probability of seeing at least 100 galaxies is greater than 50%? (Think simple!)

# %%
def n_FOV(x, expected_x_per_FOV=19.3,min_probability=0.5):
    n = min_probability*(x/expected_x_per_FOV)
    return n

# %%
mean = 19.3
min_prob = 0.5
galaxies = 100

n = n_FOV(x = galaxies,
          expected_x_per_FOV = mean,
          min_probability = min_prob)

print("Need at least {} FOV exposures".format(round(n)))

# %% [markdown]
# ## Part (b)
# Your proposal review insists that you need at least 100 galaxies and requires a 99% chance that you will succeed. How many exposures do you need now?

# %%
mean = 19.3
min_prob = 0.99
galaxies = 100

n = n_FOV(x = galaxies,
          expected_x_per_FOV = mean,
          min_probability = min_prob)

print("Need at least {} FOV exposures".format(round(n)))

# %% [markdown]
# ## Part (c)
# There is a problem, however. A "contaminating" star is one that is bright enough to saturate the image to the point that one cannot determine the properties of galaxies. The expected (mean) number of "contaminating" stars one FOV is $5.1$. Your assignment is to find enough FOVs in the sky without contaminating stars to get the needed long-term exposures. Your advisor (who did not take statistics) comes up with a simple plan. Divide the sky in sectors, each with an area of the FOV with no overlap and take a short images; a contaminating star will emerge quickly. How many sectors do you expect to examine to find enough uncontaminated FOVs?

# %%


# %% [markdown]
# ## Part (d)
# What an effort! But another problem comes up. Due to the long exposure time and scheduling limitations of the telescope, there is a limited region of the sky that can be used. It is $1000 \text{ as}$ by $1000 \text{ as}$ whereas your FOV is a $100 \text{ as}$ diameter circle. At best there are only a little over 100 such sectors. You now remember the "Find the Gap" lecture and you realize that there are many more possible exposures. You take a star map of the entire usable region. Demonstrate you can find enough open regions. Your choice as to how.
# 
# **Hint:** One way is to create a random star map using positions as $x= \text{uniform random}$ between $0$ and $1000 \text{ as}$ and $y= \text{uniform random}$ between $0$ and $1000 \text{ as}$. Find values on a $1000 \times 1000$ grid $[x,y]$ in which no stars are within $50 \text{ as}$.

# %% Q1 Part (d)

# Let's make our random x and y arrays for the random star field:
import random
N = 510 # number of stars in the 1000 as by 1000 as field of view:
x_FOV = 1000
y_FOV = 1000
x_array=[]
y_array=[]
star_array = []    # list of tuples of star-center coordinates (xi,yi)
for _ in range(N+1):
    xi = random.uniform(0,x_FOV)
    yi = random.uniform(0,y_FOV)
    x_array.append(xi)
    y_array.append(yi)
    star_array.append((xi,yi))

x_array = np.array(x_array)
y_array = np.array(y_array)
star_array = np.array(star_array)

# # This is our code that checks against the random star field for circular gaps of at least 50 arcseconds
# circ_centers = []
# R = 50 # radius of FOV
# for y in np.arange(50,951):
#     for x in np.arange(50,951):
#         # box test (see if any stars exist in 50 as box)
#         t=star_array[((star_array[::,0]>(x-R)) & ((star_array[::,0]<(x+R))))]
#         s=np.array(t[((t[::,1]>(y-R)) & ((t[::,1]<(y+R))))])
#         if s.shape[0] == 0:
#             circ_centers.append([x,y])
#         # circle test (see if any stars exist within 50 as radius of center)
#         # I started with box test because I think its faster, but maybe not, might just be extra uneeded step.
#         else:
#             s=s[((((s[::,0]-x)**2+(s[::,1]-y)**2)**(1/2))<R)]
#             if s.shape[0] == 0:
#                 circ_centers.append([x,y])
                                            
# circ_centers=np.array(circ_centers)

# %% [markdown]
# # 2. Multi-variant Uncertainties
# Stellar parallax measurements are made by measuring the relative motion of a nearby star ($S_A$) with respect that of a distant object ($S_B$) due to Earth's orbit about the Sun. All images, however, have finite resolution; even though a star should make a single point on an image the star's intensity has a finite width, often referred to a point-spread function. The probability that a photon is recorded at position $x$ is: 
# 
# \begin{equation}
# P(x) = \frac{1}{\Delta x \sqrt{2 \pi}} e^{-\frac{(x-x_0)^2}{2\Delta x^2}}
# \end{equation}

# %% [markdown]
# ## Part (a)
# In the above equation, $\Delta x$ is a property of, for example, the telescope diameter and the CCD resolution. Suppose $\Delta x = 20 \text{ pixels}$. Star $S_A$ records $10^3$ photons and $S_B$ yields $10^4$ photons. What is the $1-\sigma$ variance in the position of each of the stars (in *pixels*)?

# %%


# %% [markdown]
# ## Part (b)
# Given the above, what is the uncertainty in the distance between the stars?

# %%


# %% [markdown]
# ## Part (c)
# The calibration relating *pixels* to *mas* is $c=10 \text{ mas/pixel} \pm 0.2 \%$. Suppose $S_A$ and $S_B$ are separated by $250$ pixels. What is $d$ and the uncertainty in $d$ in *mas*? Which uncertainty ($a$, $b$, or $c$) makes the largest contribution to the uncertainty in $d$?
# 
# <center>
# <img src="stellar.PNG" width="300"/>
# </center>

# %%


# %% [markdown]
# # 3. Gamma Ray Bursts
# Gamma Ray Bursts (GRBs) were first detected in the 1960s by defense satellites that were designed to verify the nuclear test ban treaty. These detectors had little ability to determine the source direction, but enough to determine that GRBs were not from Earth.
# 
# To investigate GRBs, the next satellite-borne detectors had all-sky coverage combined with separate detectors with fields-of-view of roughly $0.04 \text{ sr}$ ($0.3 \%$ of the sky; gamma rays are difficult to focus). The GRB energy is so high that it was at thought by  many (at the time) that GRBs must be of galactic origin. GRBs are detected at a rate of about one per day.

# %% [markdown]
# ## Part (a)
# Start by dividing the sky into two equal parts; one part contains the galactic plane and the other part is the rest of the sky (which has two sub-parts, but not a concern here). Most models predicted that a majority (at least $60 \%$) of GRBs should be detected in a $2\pi \text{ sr}$ region containing the galactic plane. To test this hypothesis, assume that $60$ of GRBs are coming from the galactic plane. Estimate (no computer) how long was it expected to take the observations to show that the fraction of events in the galactic plane is $2\sigma$ higher than $50\%$, proving a galactic origin model.

# %%
galactic_plane = 60
# FOV = 0.04 sr => rate = 1 per day

# FOV = 2 pi sr <= at least 60% of GRBs per day 

# %% [markdown]
# ## Part (b)
# After roughly one year, say that 166 GRBs are detected in the galactic plane out of 320 events. Let's examine the expected value of $r_{GP}$, the ratio of the number of events from galactic plane to the number of total events. By computer, compute and plot the Gaussian probability distribution as a function of $r_{GP}$ using the mean and $\sigma$ from the above statistics. Why Gaussian? The number of events may be a bit too high to compute Poisson statistics directly. Mark the $\pm 2\sigma$ area. Compute the CDF. What is the probability that $r_{GP}$ is $60\%$ or higher?

# %%
yr1 = 166/320
mu = 60/()   # 1 GRB per day


# %% [markdown]
# ## Part (c)
# The idea that GRBs are extra-galactic was difficult to accept. However, after 9 years 2512 GRBs were detected of which close to $50\%$ came from the galactic plane, which ruled out all but the most die-hard galactic-origin theories. The next test was to determine if there was any bias in GRB direction. The sky was parsed into 300 equally-sized sectors. Suppose the histogram is: $$H = [0,1,2,8,9,25,37,37,40,36,41,23,23,8,3,3,2,1,1]$$
# Does the GRB histogram match a Poisson distribution?
# 
# **Hint:** Cut and paste $H$ into your code and plot. Overplot the expected distribution of the 300 sectors. What is the most likely number of GRBs per sector ($\mu'$)? What is $\chi_\nu^2$? How does $\chi_\nu^2$ compare with the expected distribution? What do you conclude?

# %%
H = [0,1,2,8,9,25,37,37,40,36,41,23,23,8,3,3,2,1,1]

# %% [markdown]
# ## Part (d)
# To perform another $\chi_\nu^2$ test, all of the 300 sectors are expected to have, within error, the same number of GRBs. That is, $f(x)=\mu'$, which is now well determined. We also know $\sigma ^2 = \mu'$ well. What is $\chi_\nu^2$? What do you conclude?
# 
# **Hint:** Explode $H$ into a 300-point array ($X$): $X = [0,1,2,3,3,3,3,3,3,3,3,(9 \text{   } 4 \text{s}),(25 \text{   } 5 \text{s}), \text{etc.}]$


