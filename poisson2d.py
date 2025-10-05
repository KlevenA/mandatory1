import numpy as np
import sympy as sp
import scipy.sparse as sparse


x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L # domain from 0 to L for x and y 
        self.ue = ue 
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)
        #self.px = Poisson(L, N)
        #self.py = Poisson(L, N)
        #self.create_mesh()
        



    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        N = self.N 
        self.h = self.L/N
        xi = np.linspace(0, self.L, N+1) # xi array
        yj = np.linspace(0, self.L, N+1) # yj array 
  
        self.xij, self.yij = np.meshgrid(xi, yj, indexing='ij') #
        return self.xij, self.yij
    
    

    def D2(self):
        """Return second order differentiation matrix"""
        # 2nd order diff.matrix
        # d^2u/dx^2=u(i-1)-2u(i)+u(i+1)/(L/N)^2

        N = self.N; L = self.L
        diagonal = np.ones(N+1)*(-2)
        diagonal_upanddown = np.ones(N)
        D2 = sparse.diags([diagonal_upanddown, diagonal, diagonal_upanddown], 
                          offsets=[-1, 0, 1], shape=(N+1,N+1))
        D2 = D2.tolil() # make format workable to add Ends from taylor exp.
        # D2[0,0:4] = 2,-5,4,1 
        # D2[-1,-4:] = -1,4,-5,2  
        D2 = D2/(L/N)**2 # /h^2 
        return D2
    
    def laplace(self):
        """Return vectorized Laplace operator"""
        # \nabla^2u = d^2u/dx^2+d^2u/dy^2
        dx = self.L/self.N; lap = 1./dx**2
        D2x = lap*self.D2()
        D2y = lap*self.D2()
        
        
        I = sparse.eye(self.N+1)
        return (sparse.kron(D2x, I)+sparse.kron(I,D2y)).tolil() # kroneckers


    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        Boundary = np.ones((self.xij.N+1, self.yij.N+1), dtype=bool) # make ones-matrix same size as grid
        Boundary[1:-1,1:-1] = 0 # making all points at boundary = 0
        return np.where(Boundary.ravel() ==1)[0] # get indices of all points==0
    
    
    def meshfunction(self, u):
        # in: u=fnction of choice
        # out: u as arr
        return sp.lambdify((x, y),u)(self.xij, self.yij)
       

    def assemble(self, f=None):
        """Return assembled matrix A and right hand side vector b"""
    
        A = self.laplace()
        boundary = self.get_boundary_indices()
        ##################################################### 
        f_function = sp.lambdify((x,y), self.f)
        ue_function = sp.lambdify((x,y), self.ue)
        b = f_function(self.xij, self.yij).ravel()
        
        A = A.tolil()
        for n in boundary:
            A.rows[n] = [n]
            A.data[n] = [1]
            i = n//(self.N+1)
            j = n % (self.N+1)
            b = float(ue_function(self.xij[i, j], self.yij[i, j]))
        A = A.tocsr()
        ######################################################
        for i in boundary:
            A[i] = 0 # setting all boundary-points to be zero
            A[i,i] = 1 # with the exception of the diagonal
        A = A.tocsr()
        
        b = np.zeros(self.xij, self.yij)
        b[:,:] = self.meshfunction(f)
        
        uij = self.meshfunction(self.ue)
        b.ravel()[boundary] = uij.ravel()[boundary]
        
        return A, b



    def l2_error(self, u):
        """Return l2-error norm"""
        dx = self.L/self.N
        dy = dx # since we have a uniform and square grid dx = dy 
        diff_uue_square = (u - self.meshfunction(self.ue))**2
        diff_sum = np.sum(diff_uue_square)
        return np.sqrt(diff_sum*dx*dy)


    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.h = self.L/N
        self.create_mesh(N)
            
        A, b = self.assemble(f=sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2))
        self.U = sparse.linalg.spsolve(A, b.ravel().reshape((N+1, N+1)))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        dx = self.h; dy = dx
        
        if x < 0 or x > self.L or y < 0 or y > self.L:
            raise ValueError("The coordinate is outside the domain")
        else:
            # Finding the closest point between (x,y) and (0,0)
            x_close = np.floor(x/dx)
            y_close = np.floor(y/dy)
            
            distx1 = x-x_close; distx2 = (x_close+dx)-x # calculating distance between x and x_grid_vals on both sides 
            disty1 = y-y_close; disty2 = (y_close+dy)-y # calculating distance between y and y_grid_vals on both sides 
            u = self.U
            
            # weights for each point
            p1 = np.sqrt(distx1**2+disty1**2) 
            p2 = np.sqrt(distx2**2+disty1**2) 
            p3 = np.sqrt(disty2**2+distx1**2)
            p4 = np.sqrt(distx2**2+disty2**2) 
            
            u_xy = ((p1)*u(x_close, y_close)
                    +(p2)*u(x_close+1, y_close)
                    +(p3)*u(x_close, y_close+1)
                    +(p4)*u(x_close+1, y_close+1))
        
        return u_xy


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

