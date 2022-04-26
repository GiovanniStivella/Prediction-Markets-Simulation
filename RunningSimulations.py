#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


def linear_trader (prob, price):
    
    if prob>price:
        alpha = 1
    else:
        alpha = 0
        
    return alpha


# In[5]:


def kelly_trader (prob, price):
    
    alpha = prob
    
    return alpha


# In[6]:


def fractional_kelly (risk, prob, price):
    
    alpha = risk*prob+(1-risk)*price
    
    return alpha


# In[7]:


def choose_agent(risk, prob, price, n):
    
    if n==0:
        alpha = linear_trader (prob, price)
    elif n==1:
        alpha = kelly_trader (prob, price)
    else:
        alpha = fractional_kelly (risk, prob, price)
    
    return alpha


# In[8]:


def market_function(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions):
    """  
    This function calculates wealth (w) of agents (w2=1-w1) and price (price) of Arrow security
    starting from the risk propension (risk) of agents and the probability they give to the event
    linked to Arrow security (prob), in the case of different true probabilities of an event (probability).
    Risk, prob, w and price are values between 0 and 1. w1+w2=1, as a market clearing condition.
    It returns a list of lists containing the changes in the values as the event gets repeated.
    """
    w2=1-w1
    s = np.random.choice(2, size=None, replace=True, p=[1-probability, probability])
    
    price = (w1*prob1*risk1+w2*prob2*risk2)/(w1*risk1+w2*risk2)
        
    sim_array = np.empty((3, n_of_repetitions))
    
    sim_array[0][0]=w1
    sim_array[1][0]=w2
    sim_array[2][0]=price
    
    for i in range(1,n_of_repetitions):    
        
        alpha1 = choose_agent (risk1, prob1, price, n1)
        alpha2 = choose_agent (risk2, prob2, price, n2)
        
        try:
            if s==1:
                w1*=alpha1/price    
                w2*=alpha2/price
    
            else:
                w1*=(1-alpha1)/(1-price)  
                w2*=(1-alpha2)/(1-price)
        
            w1 = w1/(w1+w2)
            w2 = 1-w1
        
        except ZeroDivisionError:
            w1 = w1
            w2 = w2
        
        sim_array[0][i]=w1
        sim_array[1][i]=w2
        sim_array[2][i]=price
        
        price=(w1*prob1*risk1+w2*prob2*risk2)/(w1*risk1+w2*risk2)
       
    return sim_array

# In[9]:


def look_at_the_market(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions):
    g = market_function(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions)
    
    x = [x for x in range(n_of_repetitions)]
    print('Wealth of first agent')
    plt.plot(x, g[0], label='Wealth of first agent',linewidth=3)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Wealth of first agent')
    plt.ylim(0,1)
    plt.savefig('graph1')
    plt.show()
    
    print('Wealth of second agent')
    plt.plot(x, g[1], label='Wealth of second agent',linewidth=3)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Wealth of second agent')
    plt.ylim(0,1)
    plt.savefig('graph2')
    plt.show()
    
    print('Price of first Arrow security')
    plt.plot(x, g[2], label='Price of first Arrow security',linewidth=3)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Price')
    plt.ylim(0,1)
    plt.axhline(y=probability, color='red', linestyle='--')
    plt.savefig('graph3')
    plt.show()


# In[10]:


def multiple_simulations(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions,n_of_simulations):
    
    values_array = np.empty((3, n_of_simulations, n_of_repetitions))
    
    random.seed(10)
    
    for i in range(n_of_simulations):
        a = market_function (risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions)
        
        for ind in range(3):     
            for n_rep in range(n_of_repetitions):
                values_array[ind][i][n_rep]=a[ind][n_rep]
                
    mean_array = values_array.mean(axis=1)
    std_array = values_array.std(axis=1)
    aggregate_array = np.concatenate((mean_array, std_array))
                
    return aggregate_array


# In[11]:


def look_at_the_simulations(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions, n_of_simulations):
    p = multiple_simulations(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions, n_of_simulations)
    x = [x for x in range(n_of_repetitions)]
    y1  = p[0]
    y2  = p[1]
    y3  = p[2]
    ci1 = 2*p[3]/(n_of_simulations**0.5)
    ci2 = 2*p[4]/(n_of_simulations**0.5)
    ci3 = 2*p[5]/(n_of_simulations**0.5)
    
    
    print('Wealth of first agent')
    plt.plot(x, y1, label='Wealth of first agent',linewidth=3)
    plt.fill_between(x, (y1-ci1), (y1+ci1), color = 'b', alpha =.1)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Wealth of first agent')
    plt.ylim(0,1)
    
    plt.savefig('graph1')
    plt.show()
    
    print('Wealth of second agent')
    plt.plot(x, y2, label='Wealth of second agent',linewidth=3)
    plt.fill_between(x, (y2-ci2), (y2+ci2), color = 'b', alpha =.1)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Wealth of second agent')
    plt.ylim(0,1)
    
    plt.savefig('graph2')
    plt.show()
    
    print('Price of first Arrow security')
    plt.plot(x, y3, label='Price of first Arrow security',linewidth=3)
    plt.fill_between(x, (y3-ci3), (y3+ci3), color = 'b', alpha =.1)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Price')
    plt.ylim(0,1)
    plt.axhline(y=probability, color='red', linestyle='--')
    plt.savefig('graph3')
    plt.show()


# In[12]:


def comparative_array(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions,n_of_simulations, levels):
    
    array = np.empty((6,levels))
    
    for i in range(levels):
        p = multiple_simulations(i/levels, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions,n_of_simulations)
        
        for ind in range(6):
            array[ind][i]=p[ind][n_of_repetitions-1]
            
    return array


# In[13]:


def comparative_look(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions,n_of_simulations, levels):
    g = comparative_array(risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions,n_of_simulations, levels)
    
    x = [x/levels for x in range(levels)]
    y1 = g[0]
    y2 = g[1]
    y3 = g[2]
    ci1 = 2*g[3]/(n_of_simulations**0.5)
    ci2 = 2*g[4]/(n_of_simulations**0.5)
    ci3 = 2*g[5]/(n_of_simulations**0.5)
    
    print('Wealth of first agent')
    plt.plot(x, y1, label='Wealth of first agent',linewidth=3)
    plt.fill_between(x, (y1-ci1), (y1+ci1), color = 'b', alpha =.1)
    plt.xlabel('Risk propensity of first agent')
    plt.ylabel('Wealth of first agent')
    plt.ylim(0,1)
    
    plt.savefig('graph1')
    plt.show()
    
    print('Wealth of second agent')
    plt.plot(x, y2, label='Wealth of second agent',linewidth=3)
    plt.fill_between(x, (y2-ci2), (y2+ci2), color = 'b', alpha =.1)
    plt.xlabel('Risk propensity of first agent')
    plt.ylabel('Wealth of second agent')
    plt.ylim(0,1)
    
    plt.savefig('graph2')
    plt.show()
    
    print('Price of first Arrow security')
    plt.plot(x, y3, label='Price of first Arrow security',linewidth=3)
    plt.fill_between(x, (y3-ci3), (y3+ci3), color = 'b', alpha =.1)
    plt.xlabel('Risk propensity of first agent')
    plt.ylabel('Price')
    plt.ylim(0,1)
    plt.axhline(y=probability, color='red', linestyle='--')
    plt.savefig('graph3')
    plt.show()


# In[14]:


print("There are two principal and useful functions:look_at_the_simulations, which show time series given all parameters, and comparative_look which shows variation of values in a certain time (t-> ∞) given different risk propensities")
print("look_at_the_simulations process these variables:risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions, n_of_simulations")
print("comparative_look process these variables: risk1, prob1, n1, risk2, prob2, n2, w1, probability, n_of_repetitions,n_of_simulations, levels")


# In[15]:


def market_for_linear(risk1, prob1, n1, risk2, prob2, n2, w1, price, probability, n_of_repetitions):
    """  
    This function calculates wealth (w) of agents (w2=1-w1) and price (price) of Arrow security
    starting from the risk propension (risk) of agents and the probability they give to the event
    linked to Arrow security (prob), in the case of different true probabilities of an event (probability).
    Risk, prob, w and price are values between 0 and 1. w1+w2=1, as a market clearing condition.
    It returns a list of lists containing the changes in the values as the event gets repeated.
    """
    w2=1-w1
    s = np.random.choice(2, size=None, replace=True, p=[1-probability, probability])
    
    sim_array = np.empty((3, n_of_repetitions+1))
        
    sim_array[0][0]=w1
    sim_array[1][0]=w2
    sim_array[2][0]=price    
    
    
    for i in range(1,n_of_repetitions):
        
        alpha1 = choose_agent (risk1, prob1, price, n1)
        alpha2 = choose_agent (risk2, prob2, price, n2)
        
        try:
            if s==1:
                w1*=alpha1/price    
                w2*=alpha2/price
    
            else:
                w1*=(1-alpha1)/(1-price)  
                w2*=(1-alpha2)/(1-price)
        
            w1 = w1/(w1+w2)
            w2 = 1-w1
        
        except ZeroDivisionError:
            w1 = w1
            w2 = w2
        
        sim_array[0][i]=w1
        sim_array[1][i]=w2
        sim_array[2][i]=price
        
        price=w1*alpha1+w2*alpha2
       
    return sim_array


# In[16]:


def look_at_linear_market(risk1, prob1, n1, risk2, prob2, n2, w1, price, probability, n_of_repetitions):
    g = market_for_linear(risk1, prob1, n1, risk2, prob2, n2, w1, price, probability, n_of_repetitions)
    
    x = [x for x in range(n_of_repetitions)]
    print('Wealth of first agent')
    plt.plot(x, g[0], label='Wealth of first agent',linewidth=3)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Wealth of first agent')
    plt.ylim(0,1)
    
    plt.savefig('graph1')
    plt.show()
    
    print('Wealth of second agent')
    plt.plot(x, g[1], label='Wealth of second agent',linewidth=3)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Wealth of second agent')
    plt.ylim(0,1)
    
    plt.savefig('graph2')
    plt.show()
    
    print('Price of first Arrow security')
    plt.plot(x, g[2], label='Price of first Arrow security',linewidth=3)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Price')
    plt.ylim(0,1)
    plt.axhline(y=probability, color='red', linestyle='--')
    plt.savefig('graph3')
    plt.show()


# In[17]:


def linear_simulations(risk1, prob1, n1, risk2, prob2, n2, w1, price, probability, n_of_repetitions, n_of_simulations):
    
    values_array = np.empty((3, n_of_simulations, n_of_repetitions))
    
    random.seed(10)
    
    for i in range(n_of_simulations):
        a = market_for_linear (risk1, prob1, n1, risk2, prob2, n2, w1, price, probability, n_of_repetitions)
        
        for ind in range(3):     
            for n_rep in range(n_of_repetitions):
                values_array[ind][i][n_rep]=a[ind][n_rep]
                
    mean_array = values_array.mean(axis=1)
    std_array = values_array.std(axis=1)
    aggregate_array = np.concatenate((mean_array, std_array))
                
    return aggregate_array


# In[18]:


def look_at_linear_simulations(risk1, prob1, n1, risk2, prob2, n2, w1, price, probability, n_of_repetitions, n_of_simulations):
    p = linear_simulations(risk1, prob1, n1, risk2, prob2, n2, w1, price, probability, n_of_repetitions, n_of_simulations)
    x = [x for x in range(n_of_repetitions)]
    y1  = p[0]
    y2  = p[1]
    y3  = p[2]
    ci1 = 2*p[3]/(n_of_simulations**0.5)
    ci2 = 2*p[4]/(n_of_simulations**0.5)
    ci3 = 2*p[5]/(n_of_simulations**0.5)
    
    
    print('Wealth of first agent')
    plt.plot(x, y1, label='Wealth of first agent',linewidth=3)
    plt.fill_between(x, (y1-ci1), (y1+ci1), color = 'b', alpha =.1)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Wealth of first agent')
    plt.ylim(0,1)
    plt.savefig('graph1')
    plt.show()
    
    
    print('Wealth of second agent')
    plt.plot(x, y2, label='Wealth of second agent',linewidth=3)
    plt.fill_between(x, (y2-ci2), (y2+ci2), color = 'b', alpha =.1)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Wealth of second agent')
    plt.ylim(0,1)
    
    plt.savefig('graph2')
    plt.show()
    
    print('Price of first Arrow security')
    plt.plot(x, y3, label='Price of first Arrow security',linewidth=3)
    plt.fill_between(x, (y3-ci3), (y3+ci3), color = 'b', alpha =.1)
    plt.xlabel('Number of repetitions')
    plt.ylabel('Price')
    plt.ylim(0,1)
    plt.axhline(y=probability, color='red', linestyle='--')
    plt.savefig('graph3')
    plt.show()


# In[19]:


print("A different function is required for linear agents, because they can act only as adaptive agents: price must be given")
print("We'll define a linear market where also linear agents can trade")
print("The important function is now look_at_linear_simulations. Its arguments are risk1, prob1, n1, risk2, prob2, n2, w1, price, probability, n_of_repetitions, n_of_simulations")


# In[ ]:




