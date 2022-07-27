# This file contains functions useful on snubber calculations
import numpy as np 
from scipy.constants import Boltzmann as kb
pi = np.pi

def parallel(x, y):
    # calculates the parallel impedance of x and y
    z = 1/(1/x + 1/y)
    return z

def get_params(c, lsq, fc=10e6):
    # calculates the snubber resistance that results in the specified cutoff frequecy for the given snubber capacitance and squid input inductance
    
    wc = 2*np.pi*fc
    
    rsq = -2/(wc*c)**2 + (wc*lsq - 1/wc/c)**2
    
    if rsq<=0:
        return np.nan
    else:
        return np.sqrt(rsq)
    
def get_zsnub(f, r, c):
    # calculates the impedance of the snubber for given resistance and capacitance
    w = 2*np.pi*f
    zsnub = r + 1/1j/w/c
    
    return zsnub
    

def get_tf(f, zsnub, lsq):
    # calculates snubber transfer function for given snubber impedance and squid input inductance
    w = 2*np.pi*f   
    zlsq = 1j*w*lsq   
    tf = (zsnub) / (zsnub + zlsq)
   
    return np.absolute(tf)

def get_fc(f, tf):
    # finds cutoff frequency from transfer function
    tf = np.absolute(tf)   
    idx = np.argmin(np.absolute(tf-1/np.sqrt(2)))   
    fc = f[idx]
   
    return fc

def get_max_tf(tf):
    # finds maximum value of snubber transfer function
    # useful for quantifying resonances from snubber
    tf = np.absolute(tf)
    return np.amax(tf)

def get_csf(f, zsnub, lsq, rtes):
    # calculates the current sharing factor for a given snubber impedance, squid input inductance, and TES operating resistance
    w = 2*np.pi*f
    zlsq = 1j*w*lsq
    csf = 1 + zlsq / parallel(zsnub, rtes)
   
    return np.absolute(csf)

def get_nei(f, zsnub, lsq, temp):
    # calculates the snubber johnson noise for given snubber impedance, squid input inductance, snubber temperature
    w = 2*np.pi*f
   
    zlsq = 1j*w*lsq
    zloop = zsnub + zlsq
   
    vn = np.sqrt(4*kb*temp*np.real(zsnub))
   
    nei = np.absolute(vn/(zsnub + zlsq)) * np.sqrt(2)
   
    return nei

def get_Rmax_from_nei(nei, T, Cs, Lsq, freq):
    # given a snubber capacitor value, a squid input inductance, a maximum allowable NEI and the temperature of the snubber,
    # calculate the maximum allowable resistance of the snubber
    
    # From notes, NEI = sqrt(4 kb T R) / sqrt(R^2 + (Xl-Xc)^2 )* sqrt(2)
    # rearranging: R^2 + 8kbT/NEI^2 * R + (Xl-Xc)^2 = 0
    # R^2 + b R + c = 0
    
    w = freq * 2 * pi # omega
    
    Xl = w * Lsq # reactance of inductor
    Xc = 1/(w*Cs) # reactance of capacitor
    
    # Bhaskara equation
    b = -8*kb*T/(nei**2)
    c = (Xl-Xc)**2
    
    delta = b**2 - 4*1*c
    Rmax1 = (-b + np.sqrt(delta) )/2
    Rmax2 = (-b - np.sqrt(delta) )/2
    #Rmax = 0
    
    return(Rmax1, Rmax2)
    
def show_plot(x_goal, x, tol=0.005):
    """ used to plot things if x is between min and max value"""
    xmin = x_goal - tol
    xmax = x_goal + tol
    do_plot = x>xmin and x<xmax
    return do_plot