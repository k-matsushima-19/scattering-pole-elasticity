#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import scipy.special
from tqdm import tqdm
from scipy import linalg
import concurrent.futures



def proc(w, f, V, **kwargs):
    return np.block([f(w, vec.T, **kwargs) for vec in V.T])

class SSM:
    def __init__(self, L, K, delta, eps, n):
        self.L = L # block SSMの列数 (計算可能な重複固有値の数の上限) L
        self.K = K # 計算可能な固有値の数の上限 = K*L
        self.delta = delta # 特異値の切り捨て
        self.eps = eps # 固有値の有無の判定基準
        self.n = n # 行列のサイズ

        if self.L > self.n:
            print("# Error: L > n")
            exit(-1)

        #
        # V = column-fullrank random matrix
        #
        fullrank = False
        while(not fullrank):
            self.V = np.matrix(np.random.rand(self.n,self.L) + 1j*np.random.rand(self.n,self.L))
            # V should not be rank deficient
            u, s, vh = np.linalg.svd(self.V, full_matrices=False)
            fullrank = s[-1]/s[0] > 1e-4

        #
        # U = column-fullrank random matrix
        #
        fullrank = False
        while(not fullrank):
            self.U = np.matrix(np.random.rand(self.n,self.L) + 1j*np.random.rand(self.n,self.L))
            # vmat should not be rank deficient
            u, s, vh = np.linalg.svd(self.U, full_matrices=False)
            fullrank = s[-1]/s[0] > 1e-4

    

    # f = A^{-1}(z)*ベクトルを計算する関数
    def run(self, f, n, rho, z0, parallel=False, max_workers=4, **kwargs):
        #
        # compute S
        #
        S = [np.zeros((self.n,self.L),dtype=complex) for k in range(2*self.K)]
        # for each integration point w
        # Ys = np.ndarray((n,self.n,self.L),dtype=complex)
        i = 0
        ws = [z0+rho*np.exp(1j*theta) for theta in np.linspace(0,2*np.pi-(2*np.pi/n),n)]

        # 並列化
        if parallel:
            #with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as excuter:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as excuter:
                # Ys = list(tqdm(excuter.map(proc, ws, [f for _ in range(n)], [self.V for _ in range(n)]), total=n))
                Ys = list(
                    tqdm(
                        excuter.map(lambda a,b,c,d: proc(a,b,c,**d), ws, [f for _ in range(n)], [self.V for _ in range(n)], [kwargs for _ in range(n)]), total=n
                        )
                    )
        else:
            # Ys = list(map(proc, ws, [f for _ in range(n)], [self.V for _ in range(n)]))
            Ys = []
            for w, f, V in tqdm(zip(ws,[f for _ in range(n)], [self.V for _ in range(n)]), total=n):
                Ys.append(proc(w,f,V, **kwargs))
            

        Ymax = 0.0
        for w, Y in zip(ws,Ys):
            # store max value
            Ymax = max(Ymax,np.max(np.abs(Y)))

            # compute S matrix for each k=0,1,..,2*K-1
            zc = (w-z0)/rho 
            for k in range(2*self.K):
                S[k] += zc * Y
                zc = zc * (w-z0)/rho

        print("# ")
        print("# SSM results")
        print("# ")
                
        # if S[0] is small, then no eigenvalue
        print("# S[0]", np.max(abs(S[0])))
        print("# max(|Y|): ", Ymax/n)
        if(np.max(abs(S[0])) < Ymax/n * self.eps): 
            print("# No eigenvalue found")
            return [], []
            

        # Compute M
        M = [self.U.H @ S[k] for k in range(2*self.K)]
        # print(M[0])
        # exit(0)

        # Hankel matrix
        H = np.zeros((self.L*self.K, self.L*self.K), dtype=complex)
        Hs = np.zeros((self.L*self.K, self.L*self.K), dtype=complex)
        jj = 0
        for j in range(self.K):
            ii = 0
            for i in range(self.K):
                H[ii:ii+self.L, jj:jj+self.L] = M[i+j]
                Hs[ii:ii+self.L, jj:jj+self.L] = M[i+j+1]

                ii += self.L

            jj += self.L

        

        # SVD of H
        u, s, vh = np.linalg.svd(H, full_matrices=False)
        nev = sum(e/s[0] > self.delta for e in s) # number of eigenvalues
        print("# Singular values: ")
        print(s)
        print("# ")

        H = H[0:nev,0:nev]
        Hs = Hs[0:nev,0:nev]
        ev, vr = linalg.eig(Hs, b=H, right=True)

        if any(abs(e)>1 for e in ev): 
            print("# Warning: Eigenvalue found outside the path")

        if 2*self.K < nev:
            print("# Error: too small K, nev = ", nev)
            exit(1)

        print("# Number of eigenvalues: ", nev)
        print("# Eigenvalues: ", [rho*zeta+z0 for zeta in ev])

        # Compute right eigenvector
        Sh = np.block([S[k] for k in range(nev)])
        Sh = Sh[:,0:nev]
        # print(Sh.shape, vr.shape)
        vectors = Sh @ vr

        # Normalize eigenvectors
        for i in range(nev):
            norm = 1 / vectors[np.argmax(np.abs(vectors[:,i])),i]
            vectors[:,i] = vectors[:,i] * norm

        return [rho*zeta+z0 for zeta in ev], vectors.T
        



