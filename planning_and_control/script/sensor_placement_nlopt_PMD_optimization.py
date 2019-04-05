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


class optimize_edgelength(object):
    def __init__(self):
        self.d1 = 1.5 # safety distance between two quads
        self.d2 = 0.5 # safety distance with the target
        self.d3 = 2.0
        self.xt = np.array([0, 0, 0]) # (x,y) values of target
        #self.time = range(20, 101, 1)
        ti  = 50; to = 100; n = 100
        self.dt = 0.01#(to-ti)/n
        self.v = 1.0
        self.time = np.linspace(ti,to,n+1) 
        self.time = [20, 30, 40, 50, 60]
        self.a = 0.766*22000 # neutron cps for a 3.82 micro curie neutron source
        self.chi = 2.77e-6 # cross section in m^2
        self.beta = 0.005833 # background intensity in cps for 0.34cpm measured by domino
        self.alpha = 1e-3
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
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        mu = [1+(self.chi*self.a/self.beta)/(2*self.chi+np.linalg.norm(i-self.xt)**2) for i in xr]
        #print mu, self.xt
        p, pfa = self.find_p(mu)
        #print 'the value of p is:', p
        if grad.size > 0 and p != 0:
            a = (p*mu[0]**(p-1)+(1-p)*(p*mu[0]**(p-1)*np.log(mu[0])+mu[0]**(p-1))-1)*self.T*self.chi*self.a
            b = (p*mu[1]**(p-1)+(1-p)*(p*mu[1]**(p-1)*np.log(mu[1])+mu[1]**(p-1))-1)*self.T*self.chi*self.a
            c = (p*mu[2]**(p-1)+(1-p)*(p*mu[2]**(p-1)*np.log(mu[2])+mu[2]**(p-1))-1)*self.T*self.chi*self.a
            
            grad[0] = -a*2*(xr[0][0]-self.xt[0])/(2*self.chi+np.linalg.norm(xr[0]-self.xt)**2)**2
            grad[1] = -a*2*(xr[0][1]-self.xt[1])/(2*self.chi+np.linalg.norm(xr[0]-self.xt)**2)**2
            grad[2] = -a*2*(xr[0][2]-self.xt[2])/(2*self.chi+np.linalg.norm(xr[0]-self.xt)**2)**2
            
            grad[3] = -b*2*(xr[1][0]-self.xt[0])/(2*self.chi+np.linalg.norm(xr[1]-self.xt)**2)**2
            grad[4] = -b*2*(xr[1][1]-self.xt[1])/(2*self.chi+np.linalg.norm(xr[1]-self.xt)**2)**2
            grad[5] = -b*2*(xr[1][2]-self.xt[2])/(2*self.chi+np.linalg.norm(xr[1]-self.xt)**2)**2
            
            grad[6] = -c*2*(xr[2][0]-self.xt[0])/(2*self.chi+np.linalg.norm(xr[2]-self.xt)**2)**2
            grad[7] = -c*2*(xr[2][1]-self.xt[1])/(2*self.chi+np.linalg.norm(xr[2]-self.xt)**2)**2
            grad[8] = -c*2*(xr[2][2]-self.xt[2])/(2*self.chi+np.linalg.norm(xr[2]-self.xt)**2)**2
        #if p != 0:
        return ((mu[0]**p+(1-p)*mu[0]**p*np.log(mu[0])-mu[0])+(mu[1]**p+(1-p)*mu[1]**p*np.log(mu[1])-mu[1])+\
            (mu[2]**p+(1-p)*mu[2]**p*np.log(mu[2])-mu[2]))*self.beta*self.T
        #else: 
            #print 'could not find the value of p for this end time T'
    
    def myconstraint1(self, x, grad): #||x1-x2||>=d1
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = -(xr[0][0]-xr[1][0])/np.linalg.norm(xr[0]-xr[1])
            grad[1] = -(xr[0][1]-xr[1][1])/np.linalg.norm(xr[0]-xr[1])
            grad[2] = -(xr[0][2]-xr[1][2])/np.linalg.norm(xr[0]-xr[1])
            grad[3] = -grad[0]; grad[4] = -grad[1]; grad[5] = -grad[2]
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0        
        return self.d1 - np.linalg.norm(xr[0]-xr[1])
    
    def myconstraint2(self, x, grad):#||x2-x3||>=d1
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = 0.0
            grad[3] = -(xr[1][0]-xr[2][0])/np.linalg.norm(xr[1]-xr[2]) 
            grad[4] = -(xr[1][1]-xr[2][1])/np.linalg.norm(xr[1]-xr[2])
            grad[5] = -(xr[1][2]-xr[2][2])/np.linalg.norm(xr[1]-xr[2])
            grad[6] = -grad[3]; grad[7] = -grad[4]; grad[8] = -grad[5]
        return self.d1 - np.linalg.norm(xr[1]-xr[2])
       
    def myconstraint3(self, x, grad):#||x3-x1||>=d1
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = (xr[2][0]-xr[0][0])/np.linalg.norm(xr[2]-xr[0])
            grad[1] = (xr[2][1]-xr[0][1])/np.linalg.norm(xr[2]-xr[0])
            grad[2] = (xr[2][2]-xr[0][2])/np.linalg.norm(xr[2]-xr[0])
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = -grad[0]; grad[7] = -grad[1]; grad[8] = -grad[2]
        return self.d1 - np.linalg.norm(xr[2]-xr[0])
        
        
    def myconstraint4(self, x, grad):#||x1-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = -(xr[0][0]-self.xt[0])/np.linalg.norm(xr[0]-self.xt)
            grad[1] = -(xr[0][1]-self.xt[1])/np.linalg.norm(xr[0]-self.xt)
            grad[2] = -(xr[0][2]-self.xt[2])/np.linalg.norm(xr[0]-self.xt)
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0
        return self.d2 - np.linalg.norm(xr[0]-self.xt)
        
    def myconstraint5(self, x, grad):#||x2-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = 0.0
            grad[3] = -(xr[1][0]-self.xt[0])/np.linalg.norm(xr[1]-self.xt)
            grad[4] = -(xr[1][1]-self.xt[1])/np.linalg.norm(xr[1]-self.xt)
            grad[5] = -(xr[1][2]-self.xt[2])/np.linalg.norm(xr[1]-self.xt)
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0
        return self.d2 - np.linalg.norm(xr[1]-self.xt)
    
    def myconstraint6(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = 0.0
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = -(xr[2][0]-self.xt[0])/np.linalg.norm(xr[2]-self.xt)
            grad[7] = -(xr[2][1]-self.xt[1])/np.linalg.norm(xr[2]-self.xt)
            grad[8] = -(xr[2][2]-self.xt[2])/np.linalg.norm(xr[2]-self.xt)
        return self.d2 - np.linalg.norm(xr[2]-self.xt)
        
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
        
    def myconstraint22(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        #aa = np.sqrt((xr[0][0]-self.xt[0])**2 + (xr[0][1]-self.xt[1])**2 + (xr[0][2]-self.xt[2])**2)
        if grad.size > 0:
            grad[0] = 2*(xr[0][0]-self.xt[0]); grad[1] = 2*(xr[0][1]-self.xt[1]); grad[2] = -2*(xr[0][2]-self.xt[2])
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0
            
        return (xr[0][0]-self.xt[0])**2 + (xr[0][1]-self.xt[1])**2 - (xr[0][2]-self.xt[2])**2

    def myconstraint23(self, x, grad):#||x3-xt||>=d2
        xr = [np.array([x[0], x[1], x[2]]), np.array([x[3], x[4], x[5]]), np.array([x[6], x[7], x[8]])]
        if grad.size > 0:
            grad[0] = 0.0; grad[1] = 0.0; grad[2] = 1.0
            grad[3] = 0.0; grad[4] = 0.0; grad[5] = 0.0
            grad[6] = 0.0; grad[7] = 0.0; grad[8] = 0.0
        return  xr[0][2] - self.xt[2] - 1.0
        
    def plot(self, xt, x1, x2, x3): 
        xx1 = []; yy1 = []; zz1 = []
        xx2 = []; yy2 = []; zz2 = []
        xx3 = []; yy3 = []; zz3 = []
        #xt.append(self.xt)
        xt = zip(*xt)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(x1)): 
            xx1.append(x1[i][0]); yy1.append(x1[i][1]); zz1.append(x1[i][2])
            xx2.append(x2[i][0]); yy2.append(x2[i][1]); zz2.append(x2[i][2])
            xx3.append(x3[i][0]); yy3.append(x3[i][1]); zz3.append(x3[i][2])


        ax.scatter(xt[0], xt[1], marker = '*', color = 'r')
        ax.scatter(xx1, yy1, zz1, marker = 'o', color='b')
        ax.scatter(xx2, yy2, zz2, marker = 'o', color='g')
        ax.scatter(xx3, yy3, zz3, marker = 'o', color='m')

        #ax.axes.set_xlim3d(-2, 2)
        #ax.axes.set_ylim3d(-2, 2)
        #ax.axes.set_zlim3d(-2, 2)
        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$', fontsize=20)
        ax.set_zlabel('$Z$', fontsize=20)
        ax.view_init(90,0)
        plt.tight_layout()
        plt.show()
            
    def optimize(self): 
       
        for i in self.time: 
            self.T = i
            v = 0.5; r = 3.0
            x1 = []; x2 = []; x3 = []; xt = []
            tt = np.linspace(0,i,101)
            for j in tt[:len(tt)]:  
                #print i
                #self.xt = np.array([r*np.cos(2*np.pi*j/self.T), r*np.sin(2*np.pi*j/self.T), 1.0])
                xt.append(self.xt)                
                #self.xt = self.xt + np.array([v*self.dt, v*self.dt, 0])
                #xt.append(list(self.xt))
                opt = nlopt.opt(nlopt.LD_MMA, 9)
                #opt.set_lower_bounds([-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')])
                opt.set_upper_bounds([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
                opt.set_lower_bounds([-50.0, -50.0, 0.0, -50.0, -50.0, 0.0, -50.0, -50.0, 0.0])
                opt.set_min_objective(self.myfunc)
                
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint1(x,grad), 1e-4)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint2(x,grad), 1e-4)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint3(x,grad), 1e-4)
                
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint4(x,grad), 1e-4)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint5(x,grad), 1e-4)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint6(x,grad), 1e-4)
    
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint7(x,grad), 1e-4)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint8(x,grad), 1e-4)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint9(x,grad), 1e-4)         
                
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint10(x,grad), 1e-4)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint11(x,grad), 1e-4)
                opt.add_inequality_constraint(lambda x,grad: self.myconstraint12(x,grad), 1e-4) 
                
                #opt.add_inequality_constraint(lambda x,grad: self.myconstraint22(x,grad), 1e-4)
                #opt.add_inequality_constraint(lambda x,grad: self.myconstraint23(x,grad), 1e-4)                 
                
                opt.set_xtol_rel(1e-3); #opt.maxeval=100000000000
                if j == 0:
                    x = opt.optimize([self.xt[0] + 0.5, self.xt[1] + 0.6, self.xt[2] + 0.4, self.xt[0] + 0.8, self.xt[1] - 0.7, \
                    self.xt[2] + 0.2, self.xt[0] -1.5, self.xt[1] + 1.1, self.xt[2] +0.1])
                    #x = opt.optimize([0.5, 0.6, 0.4, 0.8, -0.7, 0.2, -1.5, 1.1, 0.10])
                    #minf = opt.last_optimum_value()
                    x1.append([x[0], x[1], x[2]]); x2.append([x[3], x[4], x[5]]); x3.append([x[6], x[7],x[8]])
                    self.x1_i = np.array([x[0], x[1], x[2]])
                    self.x2_i = np.array([x[3], x[4], x[5]])
                    self.x3_i = np.array([x[6], x[7], x[8]])
                else:
                    x = opt.optimize([self.xt[0] + 0.5, self.xt[1] + 0.6, self.xt[2] + 0.4, self.xt[0] + 0.8, self.xt[1] - 0.7, \
                    self.xt[2] + 0.2, self.xt[0] -1.5, self.xt[1] + 1.1, self.xt[2] +0.1])
                    #x = opt.optimize([self.x1_i[0], self.x1_i[1], self.x1_i[2], self.x2_i[0], self.x2_i[1], \
                    #self.x2_i[2], self.x3_i[0], self.x3_i[1], self.x3_i[2]])

                    x1.append([x[0], x[1], x[2]]); x2.append([x[3], x[4], x[5]]); x3.append([x[6], x[7],x[8]])
                    self.x1_i = np.array([x[0], x[1], x[2]])
                    self.x2_i = np.array([x[3], x[4], x[5]])
                    self.x3_i = np.array([x[6], x[7], x[8]])
                
                #opt.set_xtol_rel(1e-3); #opt.maxeval=100000000000
                #x = opt.optimize([0.05,0.05,-0.05,0.03,0.2,-0.05]) # (x1,y1,x2,y2,x3,y3) initial guess of robot positions 
                #x = opt.optimize([self.xt+0.5, 0.1, 0.1, 0.8, -0.27, 0.2, -0.15, 0.3, 0.15])
                #x = opt.optimize([5, 1, 1, 8, -2.7, 2, -1.5, 3, 1.5])
                #minf = opt.last_optimum_value()
                #x1.append([x[0], x[1], x[2]]); x2.append([x[3], x[4], x[5]]); x3.append([x[6], x[7],x[8]])
                #print("minimum value = ", minf)
                #print 'probability of missed detection is:', np.exp(minf)
                #print("result code = ", opt.last_optimize_result())
                
            #print x1,x2,x3
            
            
            self.plot(xt, x1, x2, x3)
        


   

if __name__ == '__main__':
    f = optimize_edgelength()
    f.optimize()
