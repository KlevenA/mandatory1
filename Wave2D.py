import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import animation

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        self.N = N
        self.h = 1/self.N
        xi = np.linspace(0, 1, N+1) # xi array
        yj = np.linspace(0, 1, N+1) # yj array 
        self.xij, self.yij = np.meshgrid(xi, yj, indexing='ij') # Meshgrid 
        
        return self.xij, self.yij
        

    def D2(self, N):
        """Return second order differentiation matrix"""
        # 2nd order diff.matrix
        # d^2u/dx^2=u(i-1)-2u(i)+u(i+1)/(h)^2
        #N = self.N;
        N = self.N; h = self.h

        D2 = sparse.diags([1,-2,1], [-1,0,1],(N+1, N+1), "lil")
        D2[0,:4] = 2,-5,4,-1            # from taylor exp.
        D2[-1,-4:] = -1,4,-5,2
        D2 = D2/h**2  
        return D2
    
    @property
    def w(self):
        """Return the dispersion coefficient"""
        # w = c*pi*sqrt(mx^2+my^2) # check this
        c = self.c; kx = np.pi*self.mx; ky = np.pi*self.my
        w1 = c * np.sqrt(kx**2+ky**2)
        return w1

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        """Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        #
        self.create_mesh(N)
        
        un = np.zeros((N+1, N+1))                   # initialize u(n)
        un_1 = un.copy()                            # initialize u(n-1)
        
        # u(n-1)
        un_1[:] = sp.lambdify((x,y,t), self.ue(mx,my))(self.xij, self.yij, 0)
        
        # u(n) = u(0) + 1/2 (c^2 dt^2 \nabla^2 u(n))
        D = self.D2(N)
        un[:] = un_1 + (1/2)*((self.c*self.dt)**2*(D @ un_1 + un_1 @ D.transpose()))
        return un, un_1

    @property
    def dt(self):
        """Return the time step"""
        # dt = cfl*(min space step) / c
        
        dt = self.cfl*self.h/(self.c*np.sqrt(2))
        return dt #self.cfl*self.h/self.c 
    
    
    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        
        ue_function = sp.lambdify((x,y,t), self.ue(self.mx, self.my), "numpy")
        ue_exact = ue_function(self.xij, self.yij, t0) # value at all gridpoints at a given time t0
        #return np.sqrt((self.h**2)*(np.sum((ue_exact-u)**2)))
        return np.sqrt(np.sum((ue_exact - u)**2) * self.h**2)
    
    

    def apply_bcs(self, u):
        # boundary = 0 on all sides
        self.un1[0, :] = 0
        self.un1[:, 0] = 0
        self.un1[-1,:] = 0
        self.un1[:, -1] = 0
        
        return self.un1

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        # needed in 
        self.cfl = cfl
        self.c = c
        self.mx = mx; self.my = my
        self.create_mesh(N)
        dt = self.dt        
        h = 1/self.N
        D = self.D2(N)
        
        # u(n) stuff
        un, un_1 = self.initialize(N, mx, my)                   # u(n) and u(n-1)
        self.un1 = np.zeros((N+1, N+1))                                # empty u(n+1)
        un1=self.un1
        
        #initialize out
        solution = {}  # initializing data storage
        error = []                
        
        for n in range(1, Nt):
           
            un1[:] = 2*un - un_1 + (self.c*self.dt)**2*(D @ un + un @ D.transpose())
            un1[:] = self.apply_bcs(un1)
            un_1[:] = un
            un[:] = un1
            
            if store_data > 0 and n % store_data == 0:
                solution[n] = un.copy()
                  
                
            elif store_data == -1:
                l2err = self.l2_error(un1, (n+1)*dt)
                error.append(l2err)
                #return h, error 
            
            #un_1[:] = un                                                    # update u(n-1) to be u(n)
            #un[:] = un1.copy()                                              # update u(n) to be u(n+1)
            
        return solution if store_data > 0 else (h, error)
            
            
            
            


    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        # 2nd order diff.matrix
        # d^2u/dx^2=u(i-1)-2u(i)+u(i+1)/(L/N)^2
        N = self.N; h = self.h
        D2 = sparse.diags([1,-2,1], [-1,0,1],(N+1, N+1)).tolil()
        # change here for bc to fit neumann
        D2[0,0:4] = -2,2,0,0
        D2[-1,-4:] = 0,0,2,-2
        D2 = D2/h**2  
        return D2
        
    def ue(self, mx, my):
        # eq for standing wave Neumann 
        kx = mx*sp.pi; ky = my*sp.pi
        u_neumann = sp.cos(kx*x)*sp.cos(ky*y)*sp.cos(self.w*t)
        return u_neumann

    def apply_bcs(self, u):
        # undoing bc from former, this should be done in D2 
        return u




def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    print(abs(r[-1]-2))
    assert abs(r[-1]-2) < 1e-2 # why is error so large?

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    print(abs(r[-1]-2))
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    mx = 1
    my = mx
    cfl = 1/np.sqrt(2)
    N = 1200
    Nt = 2
    
    solD = Wave2D()
    solN = Wave2D_Neumann()
    
    h, errorD = solD(N=N, Nt=Nt, cfl=cfl, mx = mx, my=my)
    hn, errorN = solN(N=N, Nt=Nt, cfl=cfl, mx = mx, my=my)
    
    threshold = 1e-12
    print(errorD[-1])
    print(errorN[-1])
    assert errorD[-1] < threshold
    assert errorN[-1] < threshold
    


if __name__ == "__main__":
    test_convergence_wave2d(); print("convergence pass")
    
    test_convergence_wave2d_neumann(); print("convergence Neumann pass")
    
    test_exact_wave2d(); print("exact solution for Wave2D pass")
   


    


 

    
