# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:37:06 2017

@author: Samuel
"""
# change just to push to github!!!!!
"""Introduction to Bayesian Inference"""
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt

#Create a list of the number of coin tosses ('Bernoulli Trials')
number_of_trials = [0, 2, 10, 20, 50, 500]

#Conduct 500 coin tosses and output into a list of 0s and 1s
#where 0 represents a tail and 1 represents a head.
data = stats.bernoulli.rvs(.5, size=number_of_trials[-1])

#Discretize the x-axis into 100 separate plotting points
x = np.linspace(0, 1, 100)

#Loops over the number_of_trials list to continually add more coin toss data. 
#Fore each new set of data, we update our (current) prior belief to be a new
#posterior. This is carried out using what is known as the Beta-Binomial Model. 
#For the time being, we won't worry about this too much. 
for i, N in enumerate(number_of_trials):
    #Accumulate the total number of heads for this particular Bayesian update
    heads = data[:N].sum()
    
    #Create an axes subplot for each update
    ax = plt.subplot(len(number_of_trials) / 2, 2, i + 1)
    ax.set_title('%s trials, %s heads' % (N, heads))
    
    #Add labels to both axes and hide labels on y-axis
    plt.xlabel('$P(H)$, Probability of Heades')
    plt.ylabel('Density')
    if i == 0:
        plt.ylim([0.0, 2.0])
    plt.setp(ax.get_yticklabels(), visible=False)   
    
    #Create and plot a Beta Distribution to represent the posterior belief
    #in the fairness of the coin.
    y = stats.beta.pdf(x, 1 + heads, 1 + N - heads)
    plt.plot(x, y, label='observe %d tosses, \n %d heads' % (N, heads))
    plt.fill_between(x, 0, y, color='#aaaadd', alpha=.5)
    
#Expand plot to cover full width/height    
plt.tight_layout()
plt.show()


"""Bayesian Inference of a Binomial Proportion"""
import seaborn as sns
from scipy.stats import beta

sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figuresize': (8, 4)})
x = np.linspace(0, 1, 100)

params = [(.5, .5), (1,1), (4, 3), (2, 5), (6, 6)]

for p in params: 
    y = beta.pdf(x, p[0], p[1])
    plt.plot(x, y, label='$\\alpha=%s$, $\\beta=%s$' % p)

plt.xlabel('$\\theta$, Fairness')
plt.ylabel('Density')
plt.legend(title='Parameters')  

# One more commit before making a new branch!!!
  

# add to branch_new

# add new feature to branch_new