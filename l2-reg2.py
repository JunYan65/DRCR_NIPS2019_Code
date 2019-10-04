import cvxpy as cp
import numpy as np
import time

d = 5
k = 7
N = 30
C = 10

re_loss1 = np.zeros((N,k))
re_loss2 = np.zeros((N,k))
re_time = np.zeros((N,k))
mean_loss1 = np.zeros((1,k))
mean_loss2 = np.zeros((1,k))
mean_time = np.zeros((1,k))






for j in range(k):
    n = (j+1)*50
    for iter in range(N):
        X = np.random.randn(d, n)
        Y_star = sum(abs(X),1)
        Y = Y_star + 0.2* np.random.randn(1, n)
        
        start_time = time.time()
        
        Xi = cp.Variable((d,n))
        g = cp.Variable(n)
        C = cp.Variable(1)
        h = cp.Variable(n)

        cost = cp.sum(h**2)/n
        constraints = [np.max(Xi) <= np.log(n)]
        constraints = [-np.max(Xi) <= np.log(n)]
        constraints += [np.max(Xi) <= C]
        constraints += [-np.max(Xi) <= C]
        #constraints += [h >= (Y-g)]
        #constraints += [h >= (Y-g) for i in range(n)]
        constraints += [h[i] >= (Y[0][i]-g[i]) for i in range(n)]
        constraints += [h[i] >= -(Y[0][i]-g[i]) for i in range(n)]
        constraints += [
        #    g[i] - g[j] >= np.dot(Xi[:,j], X[:,i] - X[:,j]) for i in range(n) for j in range(n)
        #     g[i] - g[j] >= np.sum([Xi[j*d+k]*(X[k,i] - X[k,j]) for k in range(d)])/d for i in range(n) for j in range(n)
            g[i] - g[j] >= Xi[:,j].T@(X[:,i] - X[:,j]) for i in range(n) for j in range(n)
        ]
        prob = cp.Problem(cp.Minimize(cost),constraints)
        prob.solve()
        loss_l1 = np.mean(abs(g.value - Y_star))
        loss_l2 = np.sqrt(np.mean((g.value - Y_star)**2))
        re_loss1[iter][j] = loss_l1
        re_loss2[iter][j] = loss_l2
        re_time[iter][j] = time.time() - start_time    
        print('j:',j+1,'/',k)
        print('iter:',iter+1,'/',N)
        print('l1_loss:',loss_l1)
        print('l2_loss:',loss_l2)
        print("--- %s seconds ---" % (time.time() - start_time))
        np.savetxt("l2-loss-l1.csv", re_loss1, delimiter=",")
        np.savetxt("l2-loss-l2.csv", re_loss2, delimiter=",")
        np.savetxt("l2-time.csv", re_time, delimiter=",")
mean_loss1[0] = np.mean(re_loss1,0)
mean_loss2[0] = np.mean(re_loss2,0)
mean_time[0] = np.mean(re_time,0)
np.savetxt("l2-loss-l1-mean.csv", mean_loss1, delimiter=",")
np.savetxt("l2-loss-l2-mean.csv", mean_loss2, delimiter=",")
np.savetxt("l2-time-mean.csv", mean_time, delimiter=",")