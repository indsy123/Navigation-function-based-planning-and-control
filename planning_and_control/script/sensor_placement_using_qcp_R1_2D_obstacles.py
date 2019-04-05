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


class sensor_placement_optimization(object):
    def __init__(self): 
        """initializes variables"""
        self.xt = np.array([0, 0])
        self.nrobots = 3
        self.ndim = 2 # dimension: 2D or 3D
        self.drr = 1.5 # minimum distance between robots
        self.drt_min = 0.5 # minimum distance with the target
        self.drt_max = 1.5 # minimum distance with the target
        self.dt = 0.01
        self.T = 2
        #self.xt = np.array([0, 6.5, 1])
        #self.x1 = np.array([1, 6.7, 1.2])
        #self.x2 = np.array([0.7, 6.9, 0.8])
        #self.x3 = np.array([0.8, 6.6, 1])
        self.time_points = np.linspace(0,self.T,(self.T/self.dt)+1)
        self.workspace = np.array([0.0, 0.0, 6.0]) # square given a center half the side
        self.o1 = np.array([-4.0, -4.0, 1.0])
        self.o2 = np.array([4.0, 4.0, 1.0])
        self.O = [np.array([-4.0, -4.0, 1.0]), np.array([4.0, 4.0, 1.0])]
        self.nobstacles = 2

    def plot(self, xxt, x1, x2, x3): 

        xx1 = []; yy1 = []; zz1 = []
        xx2 = []; yy2 = []; zz2 = []
        xx3 = []; yy3 = []; zz3 = []

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
            #plt.quiver(xxt[i][0], xxt[i][1], self.xt_grad[i][0], self.xt_grad[i][1], color = 'r', scale = 40)


        plt.plot(xt[0], xt[1], marker = '*', color = 'r', markersize = 3, linestyle='--', label='Target')
        plt.plot(xx1, yy1, marker = 'o', color='b', markersize = 1, linestyle='--', label='Mav1')
        plt.plot(xx2, yy2, marker = 'o', color='g', markersize = 1, linestyle='--', label='Mav2')
        plt.plot(xx3, yy3, marker = 'o', color='m', markersize = 1, linestyle='--', label='Mav3')
        
        #print xx1, yy1
        for i in range(len(xx1)): 
            if i == 0: 
                x = [xx1[i], xx2[i], xx3[i], xx1[i]]; y = [yy1[i], yy2[i], yy3[i], yy1[i]]
                plt.plot(x, y, 'c', linewidth = 2)
                plt.plot(xt[0][i], xt[1][i], marker = '*', color = 'r', markersize = 10)
                plt.plot(xx1[i], yy1[i], marker = 'o', color='b', markersize = 6)
                plt.plot(xx2[i], yy2[i], marker = 'o', color='g', markersize = 6)
                plt.plot(xx3[i], yy3[i], marker = 'o', color='m', markersize = 6)
                plt.quiver(xxt[i][0], xxt[i][1], xt_grad[0][i], xt_grad[1][i], color = 'r', scale = 15)
                plt.quiver(xxt[i][0], xxt[i][1], xt_grad_perp[0][i], xt_grad_perp[1][i], color = 'g', scale = 15)
            elif i%30 == 0:
                x = [xx1[i], xx2[i], xx3[i], xx1[i]]; y = [yy1[i], yy2[i], yy3[i], yy1[i]]
                plt.plot(x, y, 'k', linewidth = 2)
                plt.plot(xt[0][i], xt[1][i], marker = '*', color = 'r', markersize = 10)
                plt.plot(xx1[i], yy1[i], marker = 'o', color='b', markersize = 6)
                plt.plot(xx2[i], yy2[i], marker = 'o', color='g', markersize = 6)
                plt.plot(xx3[i], yy3[i], marker = 'o', color='m', markersize = 6)
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
        r = 3.5; v = 1 # trajecotry parameter r and velocity 
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
        r1 = []; r2 = []; r3 = []; xtarget = []
        nrobot = self.nrobots
        ndim = self.ndim
        nobs = self.nobstacles
        #v = 1; r = 3.0
        opt_subopt = 0; nosol = 0; outofbound = 0
        self.xt_gradlist = []; self.xt_gradperplist = []
        for i in self.time_points: 
            #self.xt = self.xt + np.array([v*self.dt, v*self.dt])
            #self.xt = np.array([r*np.cos((i/self.T)*2*np.pi), r*np.sin((i/self.T)*2*np.pi)])
            #self.xt = np.array([r*np.cos((i/self.T)*2*np.pi), r*np.sin((2*i/self.T)*2*np.pi)])
            xt_grad, xt_gradperp = self.get_target_trajectory(i)

            xtarget.append(self.xt)
            m = Model("qcp")            
            x = m.addVars(nrobot, lb = -15, ub = 15, vtype=GRB.CONTINUOUS, name="x")
            y = m.addVars(nrobot, lb = -15, ub = 15, vtype=GRB.CONTINUOUS, name="y")
            #z = m.addVars(n, lb = 0, ub = 15, vtype=GRB.CONTINUOUS, name="z")
            #p = m.addVars(n, vtype=GRB.CONTINUOUS, name="p")
            tlist = [(p, q, r, s) for p in range(1, nrobot+1, 1) for q in range(1, nobs+1, 1) for r in range(1, ndim+1, 1) for s in range(1, 3, 1)]
            define_tuple = tuplelist(tlist)
            c = m.addVars(define_tuple, lb=0, ub=1, vtype=GRB.BINARY)
            """
            x1o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x1o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            y1o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y1o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            x1o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x1o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            y1o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y1o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            x2o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x2o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            y2o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y2o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            x2o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x2o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            y2o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y2o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
        
            x3o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x3o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            y3o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y3o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            x3o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x3o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            y3o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y3o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            """
            Q = np.eye(nrobot,nrobot)
            m.update()
            #print time.time()-t
            #Linexpr([(Q[i][j] , (x[j]-self.xt[0])) for j in range(n)])
            #obj1 = quicksum(x[i]-self.xt[0] * LinExpr([(Q[i][j], x[j]-self.xt[0]) for j in range(n)]) for i in range(n))
            w = [0.8, 0.1, 0.1]; w = w/np.linalg.norm(w)
            obj1 = quicksum(w[i]*(x[i]-self.xt[0]) * quicksum(Q[i][j] * (x[j]-self.xt[0]) for j in range(nrobot)) for i in range(nrobot))
            obj2 = quicksum(w[i]*(y[i]-self.xt[1]) * quicksum(Q[i][j] * (y[j]-self.xt[1]) for j in range(nrobot)) for i in range(nrobot))
            #obj3 = quicksum((z[i]-self.xt[2]) * quicksum(Q[i][j] * (z[j]-self.xt[2]) for j in range(n)) for i in range(n))
            obj = obj1 + obj2 #w1*obj1 + w2*obj2 #+ w3*obj3
            #print time.time()-t
            # since greater than and equal to constraints are not convex, we need to approximate them
            # see the unsolved problem 8.27 from Convex Optimization by S Boyd
            #choose a random initial position for the 3 robots
            if i == 0: 
                xr_initial = [self.xt + np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)]) for i in range(nrobot)]

                xr_initial.append(xr_initial[0]) # append first element to find a's easily
            else: 
                xr_initial = [self.x1, self.x2, self.x3]
                xr_initial.append(xr_initial[0])
            
            a = [(xr_initial[i]-xr_initial[i+1])/np.linalg.norm(xr_initial[i]-xr_initial[i+1]) for i in range(len(xr_initial)-1)]
            b = [(-xr_initial[i]+self.xt)/np.linalg.norm(-xr_initial[i]+self.xt) for i in range(len(xr_initial)-1)]
            
            m.addConstr(a[0][0]*(x[0]-x[1]) + a[0][1]*(y[0]-y[1]) >= self.drr)
            m.addConstr(a[1][0]*(x[1]-x[2]) + a[1][1]*(y[1]-y[2]) >= self.drr)
            m.addConstr(a[2][0]*(x[2]-x[0]) + a[2][1]*(y[2]-y[0]) >= self.drr)
 
            m.addConstr(b[0][0]*(-x[0]+self.xt[0]) + b[0][1]*(-y[0]+self.xt[1]) >= self.drt_min)
            m.addConstr(b[1][0]*(-x[1]+self.xt[0]) + b[1][1]*(-y[1]+self.xt[1]) >= self.drt_min)
            m.addConstr(b[2][0]*(-x[2]+self.xt[0]) + b[2][1]*(-y[2]+self.xt[1]) >= self.drt_min)

            #m.addConstr((x[0]-self.xt[0])*(x[0]-self.xt[0]) + (y[0]-self.xt[1])*(y[0]-self.xt[1])  <= self.drt_max*self.drt_max)
            #m.addConstr((x[1]-self.xt[0])*(x[1]-self.xt[0]) + (y[1]-self.xt[1])*(y[1]-self.xt[1]) <= self.drt_max*self.drt_max)
            #m.addConstr((x[2]-self.xt[0])*(x[2]-self.xt[0]) + (y[2]-self.xt[1])*(y[2]-self.xt[1])  <= self.drt_max*self.drt_max)
            
            #m.addConstr(z[0] >= self.xt[2] + 0.5)
            #m.addConstr(z[1] >= self.xt[2] + 0.5)
            #m.addConstr(z[2] >= self.xt[2] + 0.5)
            #scale = 0.25
            
            #A = np.array([[xt_gradperp[0], 0], [0, xt_gradperp[1]]]) 
            #b = self.xt
            #c = xt_grad
            #print A, b, c
            #m.addConstr(A[0][0]*x[0]-b[0]+A[1][1]*y[0]-b[1] == p[0])
            #m.addConstr((c[0]*x[0]+c[1]*y[0]) == p[1])
            #m.addConstr(p[0]*p[0] <= scale*p[1]*p[1])
            #m.addConstr(p[1] >= 0)
            #m.addConstr((A[0][0]*x[0] + b[0]) == p[0])
            #m.addConstr((A[1][1]*y[0] + b[1]) == p[1])
            #m.addConstr((c[0]*x[0]+c[1]*y[0]) == p[2])
            #m.addConstr(p[0]*p[0] + p[1]*p[1] <= scale*p[2]*p[2], "qc0")
            #m.addConstr(p[2] >= 0)

            
            # obstacle and workspace constraints
            #robot 1
            for p in range(1, nrobot+1, 1):
                for q in range(1, nobs+1, 1): 
                    #print p, q
                    m.addConstr((c[p,q,1,1] == 1) >> ((x[p-1] - self.O[q-1][0]) >= self.O[q-1][2]))
                    m.addConstr((c[p,q,1,2] == 1) >> ((x[p-1] - self.O[q-1][0]) <= -self.O[q-1][2]))
                    m.addConstr(c[p,q,1,1] + c[p,q,1,2] == 1)
                    
                    m.addConstr((c[p,q,2,1] == 1) >> ((y[p-1] - self.O[q-1][1]) >= self.O[q-1][2]))
                    m.addConstr((c[p,q,2,2] == 1) >> ((y[p-1] - self.O[q-1][1]) <= -self.O[q-1][2]))
                    m.addConstr(c[p,q,2,1] + c[p,q,2,2] == 1)
            """    
            m.addConstr((x1o1_bin1 == 1) >> ((x[0] - self.o1[0]) >= self.o1[2]))
            m.addConstr((x1o1_bin2 == 1) >> ((x[0] - self.o1[0]) <= -self.o1[2]))
            m.addConstr(x1o1_bin1 + x1o1_bin2 == 1)
            m.addConstr((y1o1_bin1 == 1) >> ((y[0] - self.o1[0]) >= self.o1[2]))
            m.addConstr((y1o1_bin2 == 1) >> ((y[0] - self.o1[0]) <= -self.o1[2]))
            m.addConstr(y1o1_bin1 + y1o1_bin2 == 1)
            m.addConstr((x1o2_bin1 == 1) >> ((x[0] - self.o2[0]) >= self.o2[2]))
            m.addConstr((x1o2_bin2 == 1) >> ((x[0] - self.o2[0]) <= -self.o2[2]))
            m.addConstr(x1o2_bin1 + x1o2_bin2 == 1)
            m.addConstr((y1o2_bin1 == 1) >> ((y[0] - self.o2[0]) >= self.o2[2]))
            m.addConstr((y1o2_bin2 == 1) >> ((y[0] - self.o2[0]) <= -self.o2[2]))
            m.addConstr(y1o2_bin1 + y1o2_bin2 == 1)            


            #robot 2
            m.addConstr((x2o1_bin1 == 1) >> ((x[1] - self.o1[0]) >= self.o1[2]))
            m.addConstr((x2o1_bin2 == 1) >> ((x[1] - self.o1[0]) <= -self.o1[2]))
            m.addConstr(x2o1_bin1 + x2o1_bin2 == 1)
            m.addConstr((y2o1_bin1 == 1) >> ((y[1] - self.o1[0]) >= self.o1[2]))
            m.addConstr((y2o1_bin2 == 1) >> ((y[1] - self.o1[0]) <= -self.o1[2]))
            m.addConstr(y2o1_bin1 + y2o1_bin2 == 1)
            m.addConstr((x2o2_bin1 == 1) >> ((x[1] - self.o2[0]) >= self.o2[2]))
            m.addConstr((x2o2_bin2 == 1) >> ((x[1] - self.o2[0]) <= -self.o2[2]))
            m.addConstr(x2o2_bin1 + x2o2_bin2 == 1)
            m.addConstr((y2o2_bin1 == 1) >> ((y[1] - self.o2[0]) >= self.o2[2]))
            m.addConstr((y2o2_bin2 == 1) >> ((y[1] - self.o2[0]) <= -self.o2[2]))
            m.addConstr(y2o2_bin1 + y2o2_bin2 == 1)
            
            #robot 3
            m.addConstr((x3o1_bin1 == 1) >> ((x[2] - self.o1[0]) >= self.o1[2]))
            m.addConstr((x3o1_bin2 == 1) >> ((x[2] - self.o1[0]) <= -self.o1[2]))
            m.addConstr(x3o1_bin1 + x3o1_bin2 == 1)
            m.addConstr((y3o1_bin1 == 1) >> ((y[2] - self.o1[0]) >= self.o1[2]))
            m.addConstr((y3o1_bin2 == 1) >> ((y[2] - self.o1[0]) <= -self.o1[2]))
            m.addConstr(y3o1_bin1 + y3o1_bin2 == 1)
            m.addConstr((x3o2_bin1 == 1) >> ((x[2] - self.o2[0]) >= self.o2[2]))
            m.addConstr((x3o2_bin2 == 1) >> ((x[2] - self.o2[0]) <= -self.o2[2]))
            m.addConstr(x3o2_bin1 + x3o2_bin2 == 1)
            m.addConstr((y3o2_bin1 == 1) >> ((y[2] - self.o2[0]) >= self.o2[2]))
            m.addConstr((y3o2_bin2 == 1) >> ((y[2] - self.o2[0]) <= -self.o2[2]))
            m.addConstr(y3o2_bin1 + y3o2_bin2 == 1)
            """
            
            m.setObjective(obj, GRB.MINIMIZE)
            #m.setObjective(sum1, GRB.MINIMIZE)
            m.write('model1.lp')
            m.setParam('OutputFlag', 0) 
            m.setParam('PSDtol', 1e-3) 
    
            m.optimize()
            
            runtime = m.Runtime
            status = m.status
            #print 'rumtime is:', runtime, status
            
            if m.status == 2 or m.status == 13: 
                opt_subopt += 1
                self.x1 = np.array([x[0].X, y[0].X])
                self.x2 = np.array([x[1].X, y[1].X])
                self.x3 = np.array([x[2].X, y[2].X])
            else: 
                nosol += 1
                self.x1 = self.x1 + np.array([0.01, 0.01])
                self.x2 = self.x2 + np.array([0.01, 0.01])
                self.x3 = self.x3 + np.array([0.01, 0.01])
            #print 'opt_subopt, nosolution:', opt_subopt, nosol
            if x[0].X < -15 or x[0].X>15: 
                print 'first robot position is out of bound'
                outofbound += 1
            #print 'no of times outof bound happend in 200 trials is:', outofbound
            r1.append(self.x1)
            r2.append(self.x2)
            r3.append(self.x3)
            #robot_positions = []
            #for i in range(n): 
            #    robot_positions.append(np.array([x[i].X, y[i].X]))
            #print time.time()-t
        self.plot(xtarget, r1, r2, r3)

        
if __name__ == '__main__':
    f = sensor_placement_optimization()
    f.optimize() 

