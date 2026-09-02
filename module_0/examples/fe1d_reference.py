"""Reference solution — 1-D FE elastostatics primer.
Bar on [0,1], EA=1, body load b(x)=x, u(0)=0, end traction F=0.3 at x=1.
Exact: u(x)=0.8x - x^3/6,  sigma(x)=EA u'(x)=0.8 - x^2/2."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

EA, L, F = 1.0, 1.0, 0.3
u_ex  = lambda x: 0.8*x - x**3/6
s_ex  = lambda x: 0.8 - x**2/2

def solve(N):
    h = L/N; x = np.linspace(0, L, N+1)
    K = np.zeros((N+1, N+1)); f = np.zeros(N+1)
    for e in range(N):
        K[np.ix_([e,e+1],[e,e+1])] += EA/h*np.array([[1,-1],[-1,1]])
        x1, x2 = x[e], x[e+1]      # consistent load for b(x)=x, linear shape fns
        f[e]   += h*(2*x1+x2)/6
        f[e+1] += h*(x1+2*x2)/6
    f[N] += F
    K, frhs = K[1:,1:], f[1:]      # essential BC u(0)=0: eliminate row/col 0
    u = np.zeros(N+1); u[1:] = np.linalg.solve(K, frhs)
    sig = EA*np.diff(u)/h          # constant per element
    return x, u, sig

print("N    max|u_h-u|_nodes   L2(u)-err     midpt-stress   TRUE energy norm")
for N in (2,4,8,16,32,64):
    x,u,sig = solve(N)
    nod = np.max(np.abs(u-u_ex(x)))
    xg = np.linspace(0,L,4001); uh = np.interp(xg,x,u)
    l2 = np.sqrt(np.trapezoid((uh-u_ex(xg))**2,xg))
    # TWO stress error measures — they converge at DIFFERENT rates, which is the point:
    #  (a) midpoint-sampled: element-midpoint stress SUPERCONVERGES -> O(h^2)
    #  (b) true energy norm: integrate (sigma_h - sigma_exact)^2 over each element -> O(h)
    xm = (x[:-1]+x[1:])/2
    mid = np.sqrt(np.sum((sig-s_ex(xm))**2*np.diff(x)))
    gq, wq = np.polynomial.legendre.leggauss(20)
    tot = 0.0
    for e in range(N):
        a, b = x[e], x[e+1]
        xg = 0.5*(b-a)*gq + 0.5*(a+b)
        tot += np.sum(wq*(sig[e]-s_ex(xg))**2)*0.5*(b-a)
    en = np.sqrt(tot)
    print(f"{N:<4} {nod:.3e}        {l2:.3e}     {mid:.3e}      {en:.3e}")

x,u,sig = solve(4)
xg = np.linspace(0,1,400)
fig,ax = plt.subplots(1,2,figsize=(9,3.6))
ax[0].plot(xg,u_ex(xg),'k-',lw=2,label='exact u'); ax[0].plot(x,u,'ro--',label='FE, N=4')
ax[0].set_xlabel('x'); ax[0].set_title('displacement'); ax[0].legend()
ax[1].plot(xg,s_ex(xg),'k-',lw=2,label='exact σ')
ax[1].step(np.repeat(x,2)[1:-1], np.repeat(sig,2), 'r-',label='FE σ (const/elem)')
ax[1].set_xlabel('x'); ax[1].set_title('stress'); ax[1].legend()
plt.tight_layout(); plt.savefig('fe1d_N4.png', dpi=130)
