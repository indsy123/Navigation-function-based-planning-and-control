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
        self.xt = np.array([0, 0, 0])
        self.no_of_sensors = 3
        self.dimension = 3 # dimension: 2D or 3D
        self.drr = 1.5 # minimum distance between robots
        self.drt_min = 0.5 # minimum distance with the target
        self.drt_max = 1.5 # minimum distance with the target
        self.dt = 0.01
        self.T = 1
        self.xt = np.array([0, 6.5, 1])
        self.x1 = np.array([1, 6.7, 1.2])
        self.x2 = np.array([0.7, 6.9, 0.8])
        self.x3 = np.array([0.8, 6.6, 1])
        self.time_points = np.linspace(0,self.T,(self.T/self.dt)+1)
        #self.workspace = np.array([0.0, 0.0, 6.0]) # square given a center half the side
        #self.o1 = np.array([-4.0, -4.0, 1.0])
        #self.o2 = np.array([4.0, 4.0, 1.0])

    def plot(self, r1, r2, r3, xt): 
        r1 = zip(*r1); r2 = zip(*r2); r3 = zip(*r3); xt = zip(*xt)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xt[0], xt[1], xt[2], marker = '*', color = 'r')
        ax.scatter(r1[0], r1[1], r1[2], marker = 'o', color='b')
        ax.scatter(r2[0], r2[1], r2[2], marker = 'o', color='g')
        ax.scatter(r3[0], r3[1], r3[2], marker = 'o', color='m')
        
        #plt.plot(xtarget[0], xtarget[1], marker = '*', color = 'r', markersize = 10)
        #plt.plot(r1[0], r1[1], marker = 'o', color = 'b', markersize = 6)
        #plt.plot(r2[0], r2[1], marker = 'o', color = 'g', markersize = 6)
        #plt.plot(r3[0], r3[1], marker = 'o', color = 'c', markersize = 6)
    
        #for i in range(len(x)): 
        #    plt.plot(x[i][0], x[i][1], marker = 'o', color = 'b', markersize = 12)
        #currentAxis = plt.gca()
        #currentAxis.add_patch(Rectangle((self.o1[0]-self.o1[2], self.o1[1]-self.o1[2]), 2*self.o1[2], 2*self.o1[2], facecolor='r', alpha=1))
        #currentAxis.add_patch(Rectangle((self.o2[0]-self.o1[2], self.o2[1]-self.o1[2]), 2*self.o2[2], 2*self.o2[2], facecolor='r', alpha=1))
        plt.xlabel('x', fontsize=24)
        plt.ylabel('y', fontsize=24)
        #plt.xlim(-self.workspace[2], self.workspace[2]); plt.ylim(-self.workspace[2], self.workspace[2])
        #plt.legend(loc=3, ncol = 3, fontsize=24)
        plt.xticks(size=16); plt.yticks(size=16)     
        plt.tight_layout()
        plt.savefig('static_qp_no_obstacle.png')
        plt.show()    

    def optimize(self):
        """ function to construct the trajectory using parallel computation"""
        t = time.time()
        r1 = []; r2 = []; r3 = []; xtarget = []
        n = self.no_of_sensors
        v = 1; r = 6.5
        opt_subopt = 0; nosol = 0
        for i in self.time_points[:len(self.time_points)]: 
            #self.xt = self.xt + np.array([v*self.dt, v*self.dt])
            self.xt = np.array([r*np.cos(2*np.pi*i/self.T), r*np.sin(2*np.pi*i/self.T), 0])
            xtarget.append(self.xt)
            m = Model("qp")            
            x = m.addVars(n, lb = -15, ub = 15, vtype=GRB.CONTINUOUS, name="x")
            y = m.addVars(n, lb = -15, ub = 15, vtype=GRB.CONTINUOUS, name="y")
            z = m.addVars(n, lb = 0, ub = 15, vtype=GRB.CONTINUOUS, name="z")
            p = m.addVars(n, vtype=GRB.CONTINUOUS, name="p")
            #x1o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x1o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #y1o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y1o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #x1o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x1o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #y1o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y1o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)

            #x2o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x2o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #y2o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y2o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #x2o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x2o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #y2o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y2o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            
            #x3o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x3o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #y3o1_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y3o1_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #x3o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); x3o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            #y3o2_bin1 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY); y3o2_bin2 = m.addVar(lb=0, ub=1, vtype=GRB.BINARY)
            Q = np.eye(n,n)
            m.update()
            print time.time()-t
            #Linexpr([(Q[i][j] , (x[j]-self.xt[0])) for j in range(n)])
            #obj1 = quicksum(x[i]-self.xt[0] * LinExpr([(Q[i][j], x[j]-self.xt[0]) for j in range(n)]) for i in range(n))
            w1 = 0.8; w2 = 0.1; w3 = 0.1; w = [w1, w2, w3]
            obj1 = quicksum(w[i]*(x[i]-self.xt[0]) * quicksum(Q[i][j] * (x[j]-self.xt[0]) for j in range(n)) for i in range(n))
            obj2 = quicksum(w[i]*(y[i]-self.xt[1]) * quicksum(Q[i][j] * (y[j]-self.xt[1]) for j in range(n)) for i in range(n))
            obj3 = quicksum(w[i]*(z[i]-self.xt[2]) * quicksum(Q[i][j] * (z[j]-self.xt[2]) for j in range(n)) for i in range(n))
            obj = obj1 + obj2 + obj3
            print time.time()-t
            # since greater than and equal to constraints are not convex, we need to approximate them
            # see the unsolved problem 8.27 from Convex Optimization by S Boyd
            #choose a random initial position for the 3 robots
            if i == 0: 
                xr_initial = [self.xt + np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)]) for i in range(n)]

                xr_initial.append(xr_initial[0]) # append first element to find a's easily
            else: 
                xr_initial = [self.x1, self.x2, self.x3]
                xr_initial.append(xr_initial[0])
            
            a = [(xr_initial[i]-xr_initial[i+1])/np.linalg.norm(xr_initial[i]-xr_initial[i+1]) for i in range(len(xr_initial)-1)]
            b = [(xr_initial[i]-self.xt)/np.linalg.norm(xr_initial[i]-self.xt) for i in range(len(xr_initial)-1)]
            
            m.addConstr(a[0][0]*(x[0]-x[1]) + a[0][1]*(y[0]-y[1]) + a[0][2]*(z[0]-z[1])>= self.drr)
            m.addConstr(a[1][0]*(x[1]-x[2]) + a[1][1]*(y[1]-y[2]) + a[1][2]*(z[1]-z[2]) >= self.drr)
            m.addConstr(a[2][0]*(x[2]-x[0]) + a[2][1]*(y[2]-y[0]) + a[2][2]*(z[2]-z[0])>= self.drr)
 
            m.addConstr(b[0][0]*(x[0]-self.xt[0]) + b[0][1]*(y[0]-self.xt[1]) + b[0][2]*(z[0]-self.xt[2])>= self.drt_min)
            m.addConstr(b[1][0]*(x[1]-self.xt[0]) + b[1][1]*(y[1]-self.xt[1]) + b[1][2]*(z[1]-self.xt[2])>= self.drt_min)
            m.addConstr(b[2][0]*(x[2]-self.xt[0]) + b[2][1]*(y[2]-self.xt[1]) + b[2][2]*(z[2]-self.xt[2])>= self.drt_min)


            m.addConstr(z[0] >= self.xt[2] + 0.1)
            m.addConstr(z[1] >= self.xt[2] + 0.1)
            m.addConstr(z[2] >= self.xt[2] + 0.1)

            m.addConstr((z[0]-self.xt[2])*(z[0]-self.xt[2]) <= p[0])
            m.addConstr((x[0]-self.xt[0])*(x[0]-self.xt[0]) + (y[0]-self.xt[1])*(y[0]-self.xt[1]) <= p[0])


            """
            # obstacle and workspace constraints
            #robot 1
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
            #m.setParam('OutputFlag', 0) 
            m.setParam('PSDtol', 1e-3) 
    
            m.optimize()
            
            runtime = m.Runtime
            status = m.status
            print 'rumtime is:', runtime, status
            
            if m.status == 2 or m.status == 13: 
                opt_subopt += 1
                self.x1 = np.array([x[0].X, y[0].X, z[0].X])
                self.x2 = np.array([x[1].X, y[1].X, z[1].X])
                self.x3 = np.array([x[2].X, y[2].X, z[2].X])
            else: 
                nosol += 1
                self.x1 = self.x1 + np.array([0.01, 0.01, 0.01])
                self.x2 = self.x2 + np.array([0.01, 0.01, 0.01])
                self.x3 = self.x3 + np.array([0.01, 0.01, 0.01])
            print opt_subopt, nosol
            r1.append(self.x1)
            r2.append(self.x2)
            r3.append(self.x3)
            #robot_positions = []
            #for i in range(n): 
            #    robot_positions.append(np.array([x[i].X, y[i].X]))
            print time.time()-t
        self.plot(r1, r2, r3, xtarget)

        
if __name__ == '__main__':
    f = sensor_placement_optimization()
    f.optimize() 


