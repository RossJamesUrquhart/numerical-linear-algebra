# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:01:26 2022

@author: bwb16179
"""

from __future__ import print_function, division
import math, sys, os, numpy as np
from numpy.random import random
from matplotlib import pyplot as plt, rcParams, animation, rc
from ipywidgets import interact, interactive, fixed
from ipywidgets.widgets import *
rc('animation', html='html5')
rcParams['figure.figsize'] = 3, 3
np.set_printoptions(precision=4, linewidth=100)

def lin(a,b,x):
    return a*x+b

a=3 # slope # usually unknown
b=8 # intercept # usually unknown

n=30
x = random(n)
y = lin(a,b,x)

plt.scatter(x,y)

def sse(y, y_pred): # SSE = Sum of Squared Errors
    return ((y-y_pred)**2).sum()
def loss(y,a,b,x): # Loss Rate
    return sse(y, lin(a,x,b))
def avg_loss(y,a,b,x): # Average Loss
    return np.sqrt(loss(y,a,b,x)/n)

a_guess=1
b_guess=1

lr=0.01 # learning rate
# d[(y-(a*x+b))**2, b] = 2 (b + a x -y)      = 2 (y_pred -y)
# d[(y-(a*x+b))**2, a] = 2 x (b + a x -y)      = x * dy/db

def upd():
    global a_guess, b_guess
    
    # make a prediction using the current weights
    y_pred = lin(a_guess, b_guess, x)
    
    # calculate the derivate of the loss
    dydb = 2 * (y_pred - y)
    dyda = x*dydb
    
    # update the weights by moving in direction of the steepest gradient (dy/db)
    a_guess -= lr*dyda.mean() # Take derivative away form guess
    b_guess -= lr*dydb.mean()
    
    
fig = plt.figure(dpi=100, figsize=(5,4))
plt.scatter(x,y)
line, = plt.plot(x, lin(a_guess, b_guess, x))
plt.close()

def animate(i):
    line.set_ydata(lin(a_guess, b_guess, x))
    for i in range(10) : upd()
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(0, 40), interval=100)
