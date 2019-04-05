# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:36:26 2018

@author: indrajeet
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy import *
from matplotlib import cm
import os
from matplotlib.patches import Polygon 
from mpl_toolkits.mplot3d import Axes3D
import time 
import nlopt
import graphics


class optimize_edgelength(object):
    def __init__(self):
        self.d1 = 1.5 # safety distance between two quads
        self.d2 = 0.5 # safety distance with the target
        self.d3 = 2.0
        self.xt = np.array([0, 0]) # (x,y) values of target
        #self.time = range(20, 101, 1)
        ti  = 50; to = 100; n = 100
        self.dt = 0.01#(to-ti)/n
        self.v = 1.0
        #self.time = np.linspace(ti,to,n+1) 
        self.time = [50]
        self.a = 0.766*22000 # neutron cps for a 3.82 micro curie neutron source
        self.chi = 2.77e-6 # cross section in m^2
        self.beta = 0.005833 # background intensity in cps for 0.34cpm measured by domino
        self.alpha = 1e-3
        #self.T = 2
        #self.dt = 0.05
    def find_p(self, _mu):
        pp = 0; self.got_p = False
        while pp < 1.0: 
            pfa_exp_list = [(j**pp - pp* j**pp * np.log(j) - 1)*self.beta*self.T for j in _mu]
            #print np.exp(sum(pfa_exp_list))
            if np.exp(sum(pfa_exp_list)) < self.alpha: 
                self.got_p = True; #_p = pp
                break
            else: 
                pp = pp + 0.01
        if self.got_p == True: 
            return pp, np.exp(sum(pfa_exp_list))
        else: 
            print 'could not find a suitable value of parameter p for this set of mu'
            return 0,0
    def myfunc(self, x, grad):
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        mu = [1+(self.chi*self.a/self.beta)/(2*self.chi+np.linalg.norm(i-self.xt)**2) for i in xr]
        #print mu, self.xt
        p, pfa = self.find_p(mu)
        #print 'the value of p is:', p
        if grad.size > 0 and p != 0:
            a = (p*mu[0]**(p-1)+(1-p)*(p*mu[0]**(p-1)*np.log(mu[0])+mu[0]**(p-1))-1)*self.T*self.chi*self.a
            b = (p*mu[1]**(p-1)+(1-p)*(p*mu[1]**(p-1)*np.log(mu[1])+mu[1]**(p-1))-1)*self.T*self.chi*self.a
            c = (p*mu[2]**(p-1)+(1-p)*(p*mu[2]**(p-1)*np.log(mu[2])+mu[2]**(p-1))-1)*self.T*self.chi*self.a
            d = (p*mu[3]**(p-1)+(1-p)*(p*mu[3]**(p-1)*np.log(mu[3])+mu[3]**(p-1))-1)*self.T*self.chi*self.a
            
            grad[0] = -a*2*(xr[0][0]-self.xt[0])/(2*self.chi+np.linalg.norm(xr[0]-self.xt)**2)**2
            grad[1] = -a*2*(xr[0][1]-self.xt[1])/(2*self.chi+np.linalg.norm(xr[0]-self.xt)**2)**2
            
            grad[2] = -b*2*(xr[1][0]-self.xt[0])/(2*self.chi+np.linalg.norm(xr[1]-self.xt)**2)**2
            grad[3] = -b*2*(xr[1][1]-self.xt[1])/(2*self.chi+np.linalg.norm(xr[1]-self.xt)**2)**2
            
            grad[4] = -c*2*(xr[2][0]-self.xt[0])/(2*self.chi+np.linalg.norm(xr[2]-self.xt)**2)**2
            grad[5] = -c*2*(xr[2][1]-self.xt[1])/(2*self.chi+np.linalg.norm(xr[2]-self.xt)**2)**2

            grad[6] = -d*2*(xr[3][0]-self.xt[0])/(2*self.chi+np.linalg.norm(xr[3]-self.xt)**2)**2
            grad[7] = -d*2*(xr[3][1]-self.xt[1])/(2*self.chi+np.linalg.norm(xr[3]-self.xt)**2)**2
            
        #return ((mu[0]**p+(1-p)*mu[0]**p*np.log(mu[0])-mu[0])+(mu[1]**p+(1-p)*mu[1]**p*np.log(mu[1])-mu[1])+\
        #    (mu[2]**p+(1-p)*mu[2]**p*np.log(mu[2])-mu[2]) + (mu[3]**p+(1-p)*mu[3]**p*np.log(mu[3])-mu[3]))*self.beta*self.T

        return ((mu[0]**p*np.log(mu[0])-mu[0]+1) + (mu[1]**p*np.log(mu[1])-mu[1]+1) +\
            (mu[2]**p*np.log(mu[2])-mu[2]+1) + (mu[3]**p*np.log(mu[3])-mu[3]+1))*self.beta*self.T
    
    def myconstraint1(self, x, grad): 
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = -(xr[0][0]-xr[1][0])/np.linalg.norm(xr[0]-xr[1])
            grad[1] = -(xr[0][1]-xr[1][1])/np.linalg.norm(xr[0]-xr[1])

            grad[2] = -grad[0]; grad[3] = -grad[1]
            
            grad[4] = 0.0; grad[5] = 0.0; grad[6] = 0.0; grad[7] = 0.0
            
        return self.d1 - np.linalg.norm(xr[0]-xr[1])
    
    def myconstraint2(self, x, grad):#||x2-x3||>=d1
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0
            grad[2] = -(xr[1][0]-xr[2][0])/np.linalg.norm(xr[1]-xr[2]) 
            grad[3] = -(xr[1][1]-xr[2][1])/np.linalg.norm(xr[1]-xr[2])
            
            grad[4] = -grad[2]; grad[5] = -grad[3]
            grad[6] = 0.0; grad[7] = 0.0
        return self.d1 - np.linalg.norm(xr[1]-xr[2])
       
    def myconstraint3(self, x, grad):#||x3-x1||>=d1
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = (xr[2][0]-xr[0][0])/np.linalg.norm(xr[2]-xr[0])
            grad[1] = (xr[2][1]-xr[0][1])/np.linalg.norm(xr[2]-xr[0])
            
            grad[2] = 0.0; grad[3] = 0.0
            grad[4] = -grad[0]; grad[5] = -grad[1]
            grad[6] = 0.0; grad[7] = 0.0
            
        return self.d1 - np.linalg.norm(xr[2]-xr[0])
        
    def myconstraint4(self, x, grad):#||x3-x1||>=d1
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = (xr[3][0]-xr[0][0])/np.linalg.norm(xr[3]-xr[0])
            grad[1] = (xr[3][1]-xr[0][1])/np.linalg.norm(xr[3]-xr[0])
            
            grad[2] = 0.0; grad[3] = 0.0
            grad[4] = 0.0; grad[5] = 0.0
            grad[6] = -grad[0]; grad[7] = -grad[1]
            
            
        return self.d1 - np.linalg.norm(xr[3]-xr[0])

    def myconstraint5(self, x, grad):#||x3-x1||>=d1
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0
            grad[2] = -(xr[1][0]-xr[3][0])/np.linalg.norm(xr[1]-xr[3])
            grad[3] = -(xr[1][1]-xr[3][1])/np.linalg.norm(xr[1]-xr[3])
            
            
            grad[4] = 0.0; grad[5] = 0.0
            grad[6] = -grad[2]; grad[7] = -grad[3]
            
            
        return self.d1 - np.linalg.norm(xr[1]-xr[3])
 
    def myconstraint6(self, x, grad):#||x3-x1||>=d1
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0
            grad[2] = 0.0; grad[3] = 0.0
            grad[4] = -(xr[2][0]-xr[3][0])/np.linalg.norm(xr[2]-xr[3])
            grad[5] = -(xr[2][1]-xr[3][1])/np.linalg.norm(xr[2]-xr[3])           
            
            grad[6] = -grad[4]; grad[7] = -grad[5]            
            
        return self.d1 - np.linalg.norm(xr[2]-xr[3]) 
        
        
    def myconstraint7(self, x, grad):#||x1-xt||>=d2
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = -(xr[0][0]-self.xt[0])/np.linalg.norm(xr[0]-self.xt)
            grad[1] = -(xr[0][1]-self.xt[1])/np.linalg.norm(xr[0]-self.xt)
            
            grad[2] = 0.0; grad[3] = 0.0
            grad[4] = 0.0; grad[5] = 0.0
            grad[6] = 0.0; grad[7] = 0.0
        return self.d2 - np.linalg.norm(xr[0]-self.xt)
        
    def myconstraint8(self, x, grad):#||x2-xt||>=d2
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0
            grad[2] = -(xr[1][0]-self.xt[0])/np.linalg.norm(xr[1]-self.xt)
            grad[3] = -(xr[1][1]-self.xt[1])/np.linalg.norm(xr[1]-self.xt)
            
            grad[4] = 0.0; grad[5] = 0.0
            grad[6] = 0.0; grad[7] = 0.0
        return self.d2 - np.linalg.norm(xr[1]-self.xt)
    
    def myconstraint9(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0
            grad[2] = 0.0; grad[3] = 0.0
            grad[4] = -(xr[2][0]-self.xt[0])/np.linalg.norm(xr[2]-self.xt)
            grad[5] = -(xr[2][1]-self.xt[1])/np.linalg.norm(xr[2]-self.xt)
            grad[6] = 0.0; grad[7] = 0.0

        return self.d2 - np.linalg.norm(xr[2]-self.xt)

    def myconstraint10(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([x[4], x[5]]), np.array([x[6], x[7]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0
            grad[2] = 0.0; grad[3] = 0.0
            grad[4] = 0.0; grad[5] = 0.0
            grad[6] = -(xr[3][0]-self.xt[0])/np.linalg.norm(xr[3]-self.xt)
            grad[7] = -(xr[3][1]-self.xt[1])/np.linalg.norm(xr[3]-self.xt)
            

        return self.d2 - np.linalg.norm(xr[3]-self.xt)
        
    """    
    def myconstraint7(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = -1.0
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0
        return self.xt[2]- xr[0][2] + 0.5

    def myconstraint8(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = 0.0
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = -1.0
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0
        return self.xt[2]- xr[1][2] + 0.5

    def myconstraint9(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = 0.0
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = -1.0
        return self.xt[2]- xr[2][2] + 0.5


    def myconstraint10(self, x, grad):#||x1-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = (xr[0][0]-self.xt[0])/np.linalg.norm(xr[0]-self.xt)
            grad[1] = (xr[0][1]-self.xt[1])/np.linalg.norm(xr[0]-self.xt)
            grad[2] = (xr[0][2]-self.xt[2])/np.linalg.norm(xr[0]-self.xt)
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0
        return np.linalg.norm(xr[0]-self.xt) - self.d3 
        
    def myconstraint11(self, x, grad):#||x2-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = 0.0
            grad[3] = (xr[1][0]-self.xt[0])/np.linalg.norm(xr[1]-self.xt)
            grad[4] = (xr[1][1]-self.xt[1])/np.linalg.norm(xr[1]-self.xt)
            grad[5] = (xr[1][2]-self.xt[2])/np.linalg.norm(xr[1]-self.xt)
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0
        return np.linalg.norm(xr[1]-self.xt) - self.d3
    
    def myconstraint12(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = 0.0
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = (xr[2][0]-self.xt[0])/np.linalg.norm(xr[2]-self.xt)
            grad[7] = (xr[2][1]-self.xt[1])/np.linalg.norm(xr[2]-self.xt)
            grad[8] = (xr[2][2]-self.xt[2])/np.linalg.norm(xr[2]-self.xt)
        return np.linalg.norm(xr[2]-self.xt) - self.d3
    """    

        
    def plot(self, xxt, x1, x2, x3, x4, i): 
        fig = plt.figure('Distance Minimization_PMD')
        color = ['k', 'y', 'c']

        if i == 0: 
            xx1 = []; yy1 = []
            xx2 = []; yy2 = []
            xx3 = []; yy3 = []
            xx4 = []; yy4 = []
            xt = zip(*xxt)
            xt_grad = zip(*self.xt_gradlist)
            xt_grad_perp = zip(*self.xt_gradperplist)
            for k in range(len(x1)): 
                xx1.append(x1[k][0]); yy1.append(x1[k][1])
                xx2.append(x2[k][0]); yy2.append(x2[k][1])
                xx3.append(x3[k][0]); yy3.append(x3[k][1])
                xx4.append(x4[k][0]); yy4.append(x4[k][1])
                #plt.quiver(xxt[i][0], xxt[i][1], self.xt_grad[i][0], self.xt_grad[i][1], color = 'r', scale = 40)
    
    
            plt.plot(xt[0], xt[1], marker = '*', color = 'r', markersize = 3, linestyle='--', label='Target')
            plt.plot(xx1, yy1, marker = 'o', color='b', markersize = 1, linestyle='--', label='Mav1')
            plt.plot(xx2, yy2, marker = 'o', color='g', markersize = 1, linestyle='--', label='Mav2')
            plt.plot(xx3, yy3, marker = 'o', color='m', markersize = 1, linestyle='--', label='Mav3')
            plt.plot(xx4, yy4, marker = 'o', color='y', markersize = 1, linestyle='--', label='Mav4')
            
            #print xx1, yy1
            for j in range(len(xx1)): 
                if j == 0: 
                    x = [xx1[j], xx2[j], xx3[j], xx4[j], xx1[j], xx3[j], xx2[j], xx4[j]]
                    y = [yy1[j], yy2[j], yy3[j], yy4[j], yy1[j], yy3[j], yy2[j], yy4[j]]
                    plt.plot(x, y, color[i], linewidth = 2)
                    plt.plot(xt[0][j], xt[1][j], marker = '*', color = 'r', markersize = 10)
                    plt.plot(xx1[j], yy1[j], marker = 'o', color='b', markersize = 6)
                    plt.plot(xx2[j], yy2[j], marker = 'o', color='g', markersize = 6)
                    plt.plot(xx3[j], yy3[j], marker = 'o', color='m', markersize = 6)
                    plt.plot(xx4[j], yy4[j], marker = 'o', color='y', markersize = 6)
                    plt.quiver(xxt[j][0], xxt[j][1], xt_grad[0][j], xt_grad[1][j], color = 'r', scale = 15)
                    plt.quiver(xxt[j][0], xxt[j][1], xt_grad_perp[0][j], xt_grad_perp[1][j], color = 'g', scale = 15)

                elif j%20 == 0:
                    x = [xx1[j], xx2[j], xx3[j], xx4[j], xx1[j], xx3[j], xx2[j], xx4[j]]
                    y = [yy1[j], yy2[j], yy3[j], yy4[j], yy1[j], yy3[j], yy2[j], yy4[j]]
                    plt.plot(x, y, color[i], linewidth = 2)
                    plt.plot(xt[0][j], xt[1][j], marker = '*', color = 'r', markersize = 10)
                    plt.plot(xx1[j], yy1[j], marker = 'o', color='b', markersize = 6)
                    plt.plot(xx2[j], yy2[j], marker = 'o', color='g', markersize = 6)
                    plt.plot(xx3[j], yy3[j], marker = 'o', color='m', markersize = 6)
                    plt.plot(xx4[j], yy4[j], marker = 'o', color='y', markersize = 6)
                    plt.quiver(xxt[j][0], xxt[j][1], xt_grad[0][j], xt_grad[1][j], color = 'r', scale = 15)
                    plt.quiver(xxt[j][0], xxt[j][1], xt_grad_perp[0][j], xt_grad_perp[1][j], color = 'g', scale = 15)

        else: 
            xx1 = []; yy1 = []
            xx2 = []; yy2 = []
            xx3 = []; yy3 = []
            xx4 = []; yy4 = []
            xt = zip(*xxt)
            xt_grad = zip(*self.xt_gradlist)
            xt_grad_perp = zip(*self.xt_gradperplist)
            for k in range(len(x1)): 
                xx1.append(x1[k][0]); yy1.append(x1[k][1]); #zz1.append(x1[i][2])
                xx2.append(x2[k][0]); yy2.append(x2[k][1]); #zz2.append(x2[i][2])
                xx3.append(x3[k][0]); yy3.append(x3[k][1]); #zz3.append(x3[i][2])
                xx4.append(x4[k][0]); yy4.append(x4[k][1])
                #plt.quiver(xxt[i][0], xxt[i][1], self.xt_grad[i][0], self.xt_grad[i][1], color = 'r', scale = 40)
    
    
            #plt.plot(xt[0], xt[1], marker = '*', color = 'r', markersize = 3, linestyle='--')
            plt.plot(xx1, yy1, marker = 'o', color='b', markersize = 0.5, linestyle='--')
            plt.plot(xx2, yy2, marker = 'o', color='g', markersize = 0.5, linestyle='--')
            plt.plot(xx3, yy3, marker = 'o', color='m', markersize = 0.5, linestyle='--')
            plt.plot(xx4, yy4, marker = 'o', color='y', markersize = 1, linestyle='--')
            
            for j in range(len(xx1)): 
                if j == 0: 
                    x = [xx1[j], xx2[j], xx3[j], xx4[j], xx1[j], xx3[j], xx2[j], xx4[j]]
                    y = [yy1[j], yy2[j], yy3[j], yy4[j], yy1[j], yy3[j], yy2[j], yy4[j]]
                    plt.plot(x, y, color[i], linewidth = 2)
                    plt.plot(xt[0][j], xt[1][j], marker = '*', color = 'r', markersize = 10)
                    plt.plot(xx1[j], yy1[j], marker = 'o', color='b', markersize = 6)
                    plt.plot(xx2[j], yy2[j], marker = 'o', color='g', markersize = 6)
                    plt.plot(xx3[j], yy3[j], marker = 'o', color='m', markersize = 6)
                    plt.plot(xx4[j], yy4[j], marker = 'o', color='y', markersize = 6)

                elif j%20 == 0:
                    x = [xx1[j], xx2[j], xx3[j], xx4[j], xx1[j], xx3[j], xx2[j], xx4[j]]
                    y = [yy1[j], yy2[j], yy3[j], yy4[j], yy1[j], yy3[j], yy2[j], yy4[j]]
                    plt.plot(x, y, color[i], linewidth = 2)
                    plt.plot(xt[0][j], xt[1][j], marker = '*', color = 'r', markersize = 10)
                    plt.plot(xx1[j], yy1[j], marker = 'o', color='b', markersize = 6)
                    plt.plot(xx2[j], yy2[j], marker = 'o', color='g', markersize = 6)
                    plt.plot(xx3[j], yy3[j], marker = 'o', color='m', markersize = 6)
                    plt.plot(xx4[j], yy4[j], marker = 'o', color='y', markersize = 6)
                    #plt.quiver(xxt[j][0], xxt[j][1], xt_grad[0][j], xt_grad[1][j], color = 'r', scale = 15)
                    #plt.quiver(xxt[j][0], xxt[j][1], xt_grad_perp[0][j], xt_grad_perp[1][j], color = 'g', scale = 15)      
            
            
        plt.legend(ncol = 2, loc = 3)    
        plt.xlabel('X (m)', fontsize = 18)
        plt.ylabel('Y (m)', fontsize = 18)
        plt.xticks(size = 18); plt.yticks(size = 18)   
        plt.tight_layout()
        plt.savefig('distance_minimization_pmd_4agents.eps')
        plt.show() 
        
    def get_target_trajectory(self, i): 
        """make a target trajectory"""
        r = 3.0; v = 1 # trajecotry paramter r and velocity 
        if i == 0: 
            self.xt_previous = np.array([r*np.cos((i/self.TT)*2*np.pi), r*np.sin((i/self.TT)*2*np.pi)]) \
            + np.array([0, -0.1])
            #+ np.array([np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)])
        else: 
            self.xt_previous = self.xt
        #self.xt = np.array([r*np.cos((i/self.T)*2*np.pi), r*np.sin((2*i/self.T)*2*np.pi)])
        self.xt = np.array([r*np.cos((i/self.TT)*2*np.pi), r*np.sin((i/self.TT)*2*np.pi)])
        g = (self.xt - self.xt_previous)/self.dt
        grad = g/np.linalg.norm(g)
        grad_perp = np.array([grad[1], -grad[0]])
        #print 'i m here', g, grad, grad_perp
        self.xt_gradlist.append(grad)
        self.xt_gradperplist.append(grad_perp)
        return grad, grad_perp
        
    def optimize(self): 
       
        for i in range(len(self.time)): 
            self.T = self.time[i]
            v = 0.5; r = 3.0
            x1 = []; x2 = []; x3 = []; x4 = []; xt = []
            self.TT = 2
            tt = np.linspace(0,self.TT,101)
            self.xt_gradlist = []; self.xt_gradperplist = []
            for j in tt:  
               
                #self.xt = self.xt + np.array([v*self.dt, 0])
                #self.xt = np.array([r*np.cos((j/tm)*2*np.pi), r*np.sin((j/tm)*2*np.pi)])
                xt_grad, xt_gradperp = self.get_target_trajectory(j)
                xt.append(self.xt) 
                opt = nlopt.opt(nlopt.LD_MMA, 8)
                #opt.set_lower_bounds([-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')])
                opt.set_upper_bounds([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])#, 50.0, 50.0, 50.0])
                opt.set_lower_bounds([-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0])#, -50.0, -50.0, 0.0])
                opt.set_min_objective(self.myfunc)
                
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint1(x,grad), 1e-8)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint2(x,grad), 1e-8)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint3(x,grad), 1e-8)
                
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint4(x,grad), 1e-8)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint5(x,grad), 1e-8)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint6(x,grad), 1e-8)
    
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint7(x,grad), 1e-8)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint8(x,grad), 1e-8)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint9(x,grad), 1e-8)   
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint10(x,grad), 1e-8)
                
                opt.set_xtol_rel(1e-3); #opt.maxeval=100000000000

                x = opt.optimize([self.xt[0]+0.2, self.xt[1]+0.8, self.xt[0]+0.4, self.xt[1]-0.7, \
                self.xt[0]-0.7, self.xt[1]-0.2, self.xt[0]-0.9, self.xt[1]+0.3])

                #minf = opt.last_optimum_value()
                x1.append([x[0], x[1]]); x2.append([x[2], x[3]]); x3.append([x[4], x[5]]); x4.append([x[6], x[7]])

            self.plot(xt, x1, x2, x3, x4, i)
        


   

if __name__ == '__main__':
    f = optimize_edgelength()
    f.optimize()
