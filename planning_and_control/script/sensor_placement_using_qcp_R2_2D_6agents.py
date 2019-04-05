import numpy as np 
from gurobipy import *
import time, scipy, operator
from functools import reduce
from scipy.linalg import block_diag
from gurobipy import *
from multiprocessing import Process, Manager, Pool
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from itertools import combinations


class sensor_placement_optimization(object):
    def __init__(self): 
        """initializes variables"""
        self.xt = np.array([3, 0])
        self.no_of_sensors = 6
        self.dimension = 2 # dimension: 2D or 3D
        self.drr = 1 # minimum distance between robots
        self.drt_min = 0.5 # minimum distance with the target
        self.drt_max = 1.5 # minimum distance with the target
        self.dt = 0.01
        self.T = 2
        self.x1 = self.xt + np.array([np.random.rand(0,1), np.random.rand(0,1)])
        self.x2 = self.xt + np.array([np.random.rand(0,1), np.random.rand(0,1)])
        self.x3 = self.xt + np.array([np.random.rand(0,1), np.random.rand(0,1)])
        self.x4 = self.xt + np.array([np.random.rand(0,1), np.random.rand(0,1)])
        self.x5 = self.xt + np.array([np.random.rand(0,1), np.random.rand(0,1)])
        self.x6 = self.xt + np.array([np.random.rand(0,1), np.random.rand(0,1)])
        self.robots = [self.x1, self.x2, self.x3, self.x4, self.x5, self.x6]
        self.time_points = np.linspace(0,self.T,(self.T/self.dt)+1)
        self.workspace = np.array([0.0, 0.0, 6.0]) # square given a center half the side
        self.o1 = np.array([-4.0, -4.0, 1.0])
        self.o2 = np.array([4.0, 4.0, 1.0])

    def plot(self, xxt, x1, x2, x3, x4, x5, x6): 

        xx1 = []; yy1 = []
        xx2 = []; yy2 = []
        xx3 = []; yy3 = []
        xx4 = []; yy4 = []
        xx5 = []; yy5 = []
        xx6 = []; yy6 = []


        xt = zip(*xxt)
        #print self.xt_grad
        xt_grad = zip(*self.xt_gradlist)
        xt_grad_perp = zip(*self.xt_gradperplist)
        #print xt_grad
        #print xt_grad_perp
        fig = plt.figure('Distance Minimization')
        #ax = fig.add_subplot(111, projection='3d')
        for i in range(len(x1)): 
            xx1.append(x1[i][0]); yy1.append(x1[i][1]); #zz1.append(x1[i][2])
            xx2.append(x2[i][0]); yy2.append(x2[i][1]); #zz2.append(x2[i][2])
            xx3.append(x3[i][0]); yy3.append(x3[i][1]); #zz3.append(x3[i][2])
            xx4.append(x4[i][0]); yy4.append(x4[i][1])
            xx5.append(x5[i][0]); yy5.append(x5[i][1])
            xx6.append(x6[i][0]); yy6.append(x6[i][1])



        plt.plot(xt[0][1:], xt[1][1:], marker = '*', color = 'r', markersize = 3, linestyle='--', label='Target')
        plt.plot(xx1[1:], yy1[1:], marker = 'o', color='b', markersize = 1, linestyle='--', label='Mav1')
        plt.plot(xx2[1:], yy2[1:], marker = 'o', color='g', markersize = 1, linestyle='--', label='Mav2')
        plt.plot(xx3[1:], yy3[1:], marker = 'o', color='m', markersize = 1, linestyle='--', label='Mav3')
        plt.plot(xx4[1:], yy4[1:], marker = 'o', color='k', markersize = 1, linestyle='--', label='Mav4')
        plt.plot(xx5[1:], yy5[1:], marker = 'o', color='c', markersize = 1, linestyle='--', label='Mav5')
        plt.plot(xx6[1:], yy6[1:], marker = 'o', color='y', markersize = 1, linestyle='--', label='Mav6')
        
        #print xx1, yy1
        for i in range(len(xx1)): 
            if i == 0: 
                x = [xx1[i],xx2[i],xx3[i],xx4[i],xx5[i],xx6[i],xx1[i],xx3[i],xx5[5],xx2[i],xx4[i],xx1[i],xx5[i],xx2[i],xx6[i],xx3[i],xx6[i],xx4[i]]
                y = [yy1[i],yy2[i],yy3[i],yy4[i],yy5[i],yy6[i],yy1[i],yy3[i],yy5[5],yy2[i],yy4[i],yy1[i],yy5[i],yy2[i],yy6[i],yy3[i],yy6[i],yy4[i]]
                #plt.plot(x, y, 'c', linewidth = 1)
                #plt.plot(xt[0][i], xt[1][i], marker = '*', color = 'r', markersize = 10)
                #plt.plot(xx1[i], yy1[i], marker = 'o', color='b', markersize = 6)
                #plt.plot(xx2[i], yy2[i], marker = 'o', color='g', markersize = 6)
                #plt.plot(xx3[i], yy3[i], marker = 'o', color='m', markersize = 6)
                #plt.plot(xx4[i], yy4[i], marker = 'o', color='k', markersize = 6)
                #plt.plot(xx5[i], yy5[i], marker = 'o', color='c', markersize = 6)
                #plt.plot(xx6[i], yy6[i], marker = 'o', color='y', markersize = 6)
                #plt.quiver(xxt[i][0], xxt[i][1], xt_grad[0][i], xt_grad[1][i], color = 'r', scale = 15)
                #plt.quiver(xxt[i][0], xxt[i][1], xt_grad_perp[0][i], xt_grad_perp[1][i], color = 'g', scale = 15)
            elif i%50 == 0:
                x = [xx1[i],xx2[i],xx3[i],xx4[i],xx5[i],xx6[i],xx1[i],xx3[i],xx5[5],xx2[i],xx4[i],xx1[i],xx5[i],xx2[i],xx6[i],xx3[i],xx6[i],xx4[i]]
                y = [yy1[i],yy2[i],yy3[i],yy4[i],yy5[i],yy6[i],yy1[i],yy3[i],yy5[5],yy2[i],yy4[i],yy1[i],yy5[i],yy2[i],yy6[i],yy3[i],yy6[i],yy4[i]]
                plt.plot(x, y, 'k', linewidth = 1)
                plt.plot(xt[0][i], xt[1][i], marker = '*', color = 'r', markersize = 10)
                plt.plot(xx1[i], yy1[i], marker = 'o', color='b', markersize = 6)
                plt.plot(xx2[i], yy2[i], marker = 'o', color='g', markersize = 6)
                plt.plot(xx3[i], yy3[i], marker = 'o', color='m', markersize = 6)
                plt.plot(xx4[i], yy4[i], marker = 'o', color='k', markersize = 6)
                plt.plot(xx5[i], yy5[i], marker = 'o', color='c', markersize = 6)
                plt.plot(xx6[i], yy6[i], marker = 'o', color='y', markersize = 6)
                plt.quiver(xxt[i][0], xxt[i][1], xt_grad[0][i], xt_grad[1][i], color = 'r', scale = 15)
                plt.quiver(xxt[i][0], xxt[i][1], xt_grad_perp[0][i], xt_grad_perp[1][i], color = 'g', scale = 15)
                

        plt.legend(ncol = 2, loc = 4)    
        plt.xlabel('X (m)', fontsize = 18)
        plt.ylabel('Y (m)', fontsize = 18)
        plt.xticks(size = 18); plt.yticks(size = 18)
   
        plt.tight_layout()
        plt.savefig('distance_minimization_qp.jpeg')
        plt.show()  
    def get_target_trajectory(self, i): 
        """make a target trajectory"""
        r = 3.0; v = 1 # trajecotry paramter r and velocity 
        if i == 0: 
            self.xt_previous = np.array([r*np.cos((i/self.T)*2*np.pi), r*np.sin((i/self.T)*2*np.pi)]) \
            + np.array([0, -0.1])
        else: 
            self.xt_previous = self.xt
        #self.xt = np.array([r*np.cos((i/self.T)*2*np.pi), r*np.sin((2*i/self.T)*2*np.pi)])
        self.xt = np.array([r*np.cos((i/self.T)*2*np.pi), r*np.sin((i/self.T)*2*np.pi)])
        
        g = (self.xt - self.xt_previous)/self.dt
        grad = g/np.linalg.norm(g)
        grad_perp = np.array([grad[1], -grad[0]])
        #print 'i m here', g, grad, grad_perp
        self.xt_gradlist.append(grad)
        self.xt_gradperplist.append(grad_perp)
        return grad, grad_perp
            
        

    def optimize(self):
        """ function to construct the trajectory using parallel computation"""
        t = time.time()
        r1 = []; r2 = []; r3 = []; r4 = []; r5 = []; r6 = []; xtarget = []
        n = self.no_of_sensors
        v = 1; r = 3.0
        opt_subopt = 0; nosol = 0; outofbound = 0
        self.xt_gradlist = []; self.xt_gradperplist = []
        for d in self.time_points: 
            #self.xt = self.xt + np.array([v*self.dt, v*self.dt])
            #self.xt = np.array([r*np.cos((i/self.T)*2*np.pi), r*np.sin((i/self.T)*2*np.pi)])
            #self.xt = np.array([r*np.cos((i/self.T)*2*np.pi), r*np.sin((2*i/self.T)*2*np.pi)])
            xt_grad, xt_gradperp = self.get_target_trajectory(d)

            xtarget.append(self.xt)
            m = Model("qcp")            
            x = m.addVars(n, lb = -15, ub = 15, vtype=GRB.CONTINUOUS, name="x")
            y = m.addVars(n, lb = -15, ub = 15, vtype=GRB.CONTINUOUS, name="y")
            #z = m.addVars(n, lb = 0, ub = 15, vtype=GRB.CONTINUOUS, name="z")

            Q = np.eye(n,n)
            m.update()
            #print time.time()-t
            #Linexpr([(Q[i][j] , (x[j]-self.xt[0])) for j in range(n)])
            #obj1 = quicksum(x[i]-self.xt[0] * LinExpr([(Q[i][j], x[j]-self.xt[0]) for j in range(n)]) for i in range(n))
            w = [0.99, 0.1, 0.1, 0.1, 0.1, 0.1]; w = w/np.linalg.norm(w)
            obj1 = quicksum(w[i]*(x[i]-self.xt[0]) * quicksum(Q[i][j] * (x[j]-self.xt[0]) for j in range(n)) for i in range(n))
            obj2 = quicksum(w[i]*(y[i]-self.xt[1]) * quicksum(Q[i][j] * (y[j]-self.xt[1]) for j in range(n)) for i in range(n))
            #obj3 = quicksum((z[i]-self.xt[2]) * quicksum(Q[i][j] * (z[j]-self.xt[2]) for j in range(n)) for i in range(n))
            obj = obj1 + obj2 #w1*obj1 + w2*obj2 #+ w3*obj3
            if d == 0.0: 
                xr_getb = [self.xt + np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)]) for i in range(n)]
            else: 
                xr_getb= self.robots
            
            b = [(xr_getb[i]-self.xt)/np.linalg.norm(xr_getb[i]-self.xt) for i in range(n)]
            m.addConstr(b[0][0]*(x[0]-self.xt[0]) + b[0][1]*(y[0]-self.xt[1]) >= self.drt_min)
            m.addConstr(b[1][0]*(x[1]-self.xt[0]) + b[1][1]*(y[1]-self.xt[1]) >= self.drt_min)
            m.addConstr(b[2][0]*(x[2]-self.xt[0]) + b[2][1]*(y[2]-self.xt[1]) >= self.drt_min)
            m.addConstr(b[3][0]*(x[3]-self.xt[0]) + b[3][1]*(y[3]-self.xt[1]) >= self.drt_min)  
            m.addConstr(b[4][0]*(x[4]-self.xt[0]) + b[4][1]*(y[4]-self.xt[1]) >= self.drt_min)
            m.addConstr(b[5][0]*(x[5]-self.xt[0]) + b[5][1]*(y[5]-self.xt[1]) >= self.drt_min)
 
      
             
            
            if d == 0.0: 
                comb = combinations(range(n), 2)
                for k in comb:
                    #print k, k[0], k[1]
                    s1 = self.xt + np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
                    s2 = self.xt + np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
                    a = (s1-s2)/np.linalg.norm(s1-s2)

                    m.addConstr(a[0]*(x[k[0]]-x[k[1]]) + a[1]*(y[k[0]]-y[k[1]]) >= self.drr)
            else: 
                comb = combinations(range(n), 2)
                for k in comb: 
   
                    s1 = self.robots[k[0]]; s2 = self.robots[k[1]]
                    a = (s1-s2)/np.linalg.norm(s1-s2)
                    m.addConstr(a[0]*(x[k[0]]-x[k[1]]) + a[1]*(y[k[0]]-y[k[1]]) >= self.drr)                 
            
            
            m.setObjective(obj, GRB.MINIMIZE)
            #m.setObjective(sum1, GRB.MINIMIZE)
            m.write('model_4agents.lp')
            m.setParam('OutputFlag', 0) 
            m.setParam('PSDtol', 1e-3) 
    
            m.optimize()
            
            runtime = m.Runtime
            status = m.status
            print 'rumtime is:', runtime, status
            
            if m.status == 2 or m.status == 13: 
                opt_subopt += 1
                self.x1 = np.array([x[0].X, y[0].X])
                self.x2 = np.array([x[1].X, y[1].X])
                self.x3 = np.array([x[2].X, y[2].X])
                self.x4 = np.array([x[3].X, y[3].X])
                self.x5 = np.array([x[4].X, y[4].X])
                self.x6 = np.array([x[5].X, y[5].X])
            else: 
                nosol += 1
                self.x1 = self.x1 + np.array([0.01, 0.01])
                self.x2 = self.x2 + np.array([0.01, 0.01])
                self.x3 = self.x3 + np.array([0.01, 0.01])
                self.x4 = self.x4 + np.array([0.01, 0.01])
                self.x5 = self.x5 + np.array([0.01, 0.01])
                self.x6 = self.x6 + np.array([0.01, 0.01])
            #print 'opt_subopt, nosolution:', opt_subopt, nosol
            self.robots = [self.x1, self.x2, self.x3, self.x4, self.x5, self.x6]
            #print self.robots
            if x[0].X < -15 or x[0].X>15: 
                print 'first robot position is out of bound'
                outofbound += 1
            print 'no of times outof bound happend in 200 trials is:', outofbound
            r1.append(self.x1)
            r2.append(self.x2)
            r3.append(self.x3)
            r4.append(self.x4)
            r5.append(self.x5)
            r6.append(self.x6)
            #robot_positions = []
            #for i in range(n): 
            #    robot_positions.append(np.array([x[i].X, y[i].X]))
            #print time.time()-t
        self.plot(xtarget, r1, r2, r3, r4, r5, r6)

        
if __name__ == '__main__':
    f = sensor_placement_optimization()
    f.optimize() 


