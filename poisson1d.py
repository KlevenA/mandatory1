import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import implemented_function

x=sp.Symbol('x')
# 1D solver to use in 2D case to simplify a bit for myself... 

class Poisson: 
    # for the 1D case 
    # u''(x)= f(x) for x in [0,L], u(0)=a, u(L) =b, a,b=#

    def __init__(self, L=1, N=None):
        self.L=L
        self.N=N
        self.ue=None
        if isinstance(N,int):
            self.mesh_it_up(N)

    def Diff2(self):
        # 2nd order diff.matrix
        D = sparse.diags([1,-2,1], [-1,0,1],(self.N+1,self.N+1),'lil')
        D[0,:4] = 2,-5,4,1
        D[-1,-4] = -1,4,-5,2 # 
        D = D/self.dx**2

        return D 
    
    def assemble(self, bc=(0,0), f=None):
        # assemble matrix and rs vector 
        # in: bc=boundary cond. at L0 and Lx
        #     f=rs of sympy func.
        # out: A=sparce mat
        #      b=1D arr. rs vec.

        D = self.Diff2()
        D[0,:4] = 1, 0, 0, 0
        D[-1,-4] = 0, 0, 0, 1
        b=np.zeros(self.N+1) # initialize b
        b[1:-1]=sp.lambdify(x,f)(self.x[1:-1])
        b[0] = bc[0]; b[-1]=bc[1] # assign b(0:1)
        return D.tocsr(), b
    
    def mesh_it_up(self, N):
        # Discretize array from 0 to L w/ equal dx

        # in: N=number of intervals 
        # out: x=discrete arr. 

        self.N = N
        self.dx = self.L/N
        self.x = np.linspace(0,self.L,self.N+1) # x from 0 to L in N steps (+1)
        return self.x
    
    def __call__(self, N, bc=(0,0), f=implemented_function('f', lambda x: 2)(x)):
        #
        # in: N=# of steps 
        #     bc= bounary cond. at L0 and Lx
        #     f= rs as sympy func
        # out: solution as np.arr

        self.mesh_it_up(N)
        A,b=self.assemble(bc=bc, f=f)
        return sparse.linalg.spsolve(A,b)
    
    def l2_error(self, u, ue):
        # find l2 error
        # in: u = numerical solution
        #     ue = analytical sol
        # out: l2_err as number 
        u_error=sp.lambdify(x,ue)(self.x)
        return np.sqrt(self.dx*np.sum((u_error-u)**2))
    
    def convergence_rates(self, m=6):
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0, f=sp.diff(self.ue, x, 2), bc=(self.ue.subs(x, 0), 
                        self.ue.subs(x, self.L)))
            E.append(self.l2_error(u, self.ue))
            h.append(self.dx)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

def test_poisson():
    sol = Poisson(1)
    sol.ue = sp.exp(4*sp.cos(x))
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2, r

if __name__ == '__main__':
    L = 2
    sol = Poisson(L=L)
    ue = sp.exp(4*sp.cos(x))
    #ue = x**2
    bc = (ue.subs(x, 0), ue.subs(x, L))
    u = sol(100, bc=bc, f=sp.diff(ue, x, 2))
    print('Manufactured solution: ', ue)
    print(f'Boundary conditions: u(0)={bc[0]:2.4f}, u(L)={bc[1]:2.2f}')
    print(f'Discretization: N = {sol.N}')
    print(f'L2-error {sol.l2_error(u, ue)}')
