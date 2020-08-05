import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

class Plotter:
    def __init__(self, state, save_trajectory=True):
        self.state = state
        self._save_trajectory = save_trajectory
        if self._save_trajectory == True:
            self._trajectory = [self.state]

        def update(self):
            pass

    def plot_trajectory(self, logscale = False, title = "Prediction Trajectory", xlabel="iteration", ylabel="Prediction"):
        if self._save_trajectory==False:
            print("trajectory not saved")
        else:
            xvals = range(len(self._trajectory))
            if logscale == True:
                plt.loglog(xvals, self._trajectory)
            else:
                plt.plot(xvals,self._trajectory)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

    def plot_error(self, truth, logscale = False, title = "Absolute Error", xlabel = "iteration", ylabel = "error"):
        if self._save_trajectory==True:
            num_iters = len(self._trajectory)
            abs_error = np.abs(self._trajectory - np.ones(num_iters)*truth)
            if logscale == True:
                plt.loglog(range(num_iters), abs_error)
            else:
                plt.plot(range(num_iters), abs_error)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        else:
            print("The absolute error is ",np.abs(truth - self.u))


class ThreeDVAR(Plotter):
    def __init__(self, a, σ, α, u, save_trajectory = False):
        super().__init__(u, save_trajectory)
        self.a = a
        self.σ = σ
        self.u = u
        self.κ = a*σ / (a**2 * σ + α)

    def update(self, y):
        self.u = self.κ*y + (1-self.κ*self.a)*self.u
        if self._save_trajectory == True:
            self._trajectory.append(self.u)

class IterateAveraged3DVAR(ThreeDVAR):
    def __init__(self, a, σ, α, u, save_trajectory = False):
        super().__init__(a, σ, α, u, save_trajectory)
        self.ubar = u
        self.num_iters = 0

    def update(self, y):
        self.num_iters += 1
        self.u = self.κ*y + (1-self.κ*self.a)*self.u
        self.ubar = (1/self.num_iters)*self.u + ((self.num_iters-1)/self.num_iters)*self.ubar
        if self._save_trajectory == True:
            self._trajectory.append(self.ubar)


class Kalman(Plotter):
    def __init__(self, a, c, γ, m, save_trajectory=False):
        super().__init__(m, save_trajectory)
        self.a = a
        self.c = c
        self.γ = γ
        self.m = m
        self.k = 0

    def update(self, y):
        self.k = self.c * self.a * ((self.a * self.c * self.a) + self.γ**2)**(-1)
        self.c = (1-self.k*a)*self.c
        self.m = self.k*y + (1-self.k * self.a)*self.m
        if self._save_trajectory == True:
            self._trajectory.append(self.m)


#set seed
rand.seed(1)

#problem parameters
u_true = 0; γ = 100; a=5;

#variance for 3DVar and IterateAveraged3DVAR, Initial variance for Kalman
σ=2;c=2

#strength of regularization
α=1

#inital guesses
u=10
m=10


sim_3DVAR = ThreeDVAR(a, σ, α, u, save_trajectory=True)
sim_ItAv3DVAR = IterateAveraged3DVAR(a, σ, α, u, save_trajectory=True)
sim_Kalman = Kalman(a, c, γ, m, save_trajectory=True)

#main loop
num_samples = 10**2
for k in range(num_samples):

    #generate data
    η = rand.normal(0, γ)
    y = a*u_true + η

    #update
    sim_3DVAR.update(y)
    sim_ItAv3DVAR.update(y)
    sim_Kalman.update(y)


plt.figure(1)
sim_Kalman.plot_error(u_true)
sim_3DVAR.plot_error(u_true)
sim_ItAv3DVAR.plot_error(u_true)
plt.yscale("log")
plt.xscale("linear")
plt.legend(["Kalman","3DVAR","IterateAveraged3DVAR"])
plt.show()

plt.figure(2)
sim_Kalman.plot_trajectory(u_true)
sim_3DVAR.plot_trajectory(u_true)
sim_ItAv3DVAR.plot_trajectory(u_true)
plt.yscale("linear")
plt.xscale("linear")
plt.legend(["Kalman","3DVAR","IterateAveraged3DVAR"])
plt.show()
