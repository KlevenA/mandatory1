import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.interpolate import interpn

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
        


    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = self.L/N
        self.xi = np.linspace(0, self.L, N+1) # xi array
        self.yj = np.linspace(0, self.L, N+1) # yj array 
  
        self.xij, self.yij = np.meshgrid(self.xi, self.yj, indexing='ij') 
        
        return self.xij, self.yij

    
    

    def D2(self):
        """Return second order differentiation matrix"""
        # 2nd order diff.matrix
        N = self.N; h = self.h

        D2 = sparse.diags([1,-2,1], [-1,0,1],(N+1, N+1)).tolil()
        D2[0,:4] = 2,-5,4,-1            # from taylor exp.
        D2[-1,-4:] = -1,4,-5,2
        D2 = D2/h**2  
        return D2
        
    
    def laplace(self):
        """Return vectorized Laplace operator"""
        # \nabla^2u = d^2u/dx^2+d^2u/dy^2
        D2 = self.D2()
        N = self.N
        laplace_vec =(sparse.kron(D2, sparse.eye(N+1))+sparse.kron(sparse.eye(N+1), D2))
        return laplace_vec


    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        boundary = np.ones((self.N+1, self.N+1), dtype=bool) # make ones-matrix same size as grid
        boundary[1:-1,1:-1] = 0 # making all points at boundary = 0
        boundary = np.where(boundary.ravel() ==1)[0] # get indices of all points==0
        return boundary 
       

    def assemble(self, f=None):
        """Return assembled matrix A and right hand side vector b"""

        A = self.laplace()
        A = A.tolil()
        f = sp.lambdify((x,y), self.f)(self.xij, self.yij)
        b = f.ravel()
        
        boundary = self.get_boundary_indices()
        
        # Dirichlet bc
        for i in boundary:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        b[boundary] = sp.lambdify((x,y), self.ue)(self.xij, self.yij).ravel()[boundary]
                      
        return A, b
        
        
    def l2_error(self, u):
        """Return l2-error norm"""
        ue_function = sp.lambdify((x,y), self.ue, "numpy")
        ue_exact = ue_function(self.xij, self.yij)
        return np.sqrt((self.h**2)*(np.sum((ue_exact-u)**2)))


    def __call__(self, N):
        """Solve Poisson's equation.
        in: N, The number of uniform intervals in each direction

        out: U, The solution as a Numpy array

        """
        self.N = N
        self.h = self.L/N
        self.create_mesh(N)
        self.get_boundary_indices()
        
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.ravel()).reshape((N+1,N+1))
            
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
        in: x, y : The coordinates for evaluation
        out The value of u(x, y)

        """
        
        if x < 0 or x > self.L or y < 0 or y > self.L:
            raise ValueError("The coordinate is outside the domain")
        else:

            xi = np.linspace(0, self.L, self.N + 1)
            yi = np.linspace(0, self.L, self.N + 1)
        
            # Interpolate 
            uxy = interpn((xi, yi), self.U, np.array([[x, y]]), method="linear")[0]
            return uxy

    

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
    threshold = 1e-3
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < threshold
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < threshold



        
if __name__ == "__main__":
    test_convergence_poisson2d()
    print("convergence pass")
    test_interpolation()
    print("interpolation pass")