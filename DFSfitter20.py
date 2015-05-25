#!/usr/bin/env python
# Version 19, compute Jacobian matrix
import numpy as np
from numpy import exp, power
import sys
from scipy.io import loadmat
from scipy.optimize import newton_krylov
from scipy.linalg import solve, inv, toeplitz, eigvals
from ProgressBar import *
import scipy.optimize
import os

'''
chang.1166@mbi.osu.edu

Take in trajectories of shape
0          x0         x0    ...
dt         x1         x1
2dt            ...
3dt
modes:
exact
exactFp
approx
approxF
'''

class DFSfitter(object):
    def __init__(self,K=.1,V=20,bins=200,D0=1,x0=2,mode="approx",upperbound=20):
        self.K=K
        self.V=V
        self.bins = bins
        self.D0 = D0
        self.x0 = x0
        self.upperbound = upperbound # domain within to solve. Stuff outside of this domain will be ignored
        self.Funreg = np.zeros(bins) # unregularized peicewise constant MLE
        self.Dunreg = np.zeros(bins) # unregularized peicewise constant MLE
        self.mode = mode # or "exact" or "single" for exact inference on traj at a time
        self.dy = 1.0*(self.upperbound - self.x0)/self.bins
        #self.divisions = self.minpos + self.dy*np.arange(self.bins)
        self.divisions = self.x0 + self.dy*np.arange(self.bins) - self.dy/2.
        self.midpoints = self.divisions #+ self.dy/2
        # coefficient matrices for the processed data
        self.fO1 = np.zeros(self.bins)
        self.fOfD = np.zeros((self.bins,3,3))
        self.fOD = np.zeros((self.bins,3))
        self.fODp = np.zeros((self.bins,3))
        self.gO11 = np.zeros(self.bins) # M
        self.gO12 = np.zeros(self.bins) # Mp
        self.gOD1 = np.zeros((self.bins,3)) # MD terms
        self.gOD2 = np.zeros((self.bins,3)) # M'D terms
        self.gODp1 = np.zeros((self.bins,3)) # MD'
        self.gODp2 = np.zeros((self.bins,3)) # M'D'
        self.gODf1 = np.zeros((self.bins,3,3)) # MDf
        self.gODf2 = np.zeros((self.bins,3,3)) # M'Df
        self.gODinv = np.zeros((self.bins,3)) # M/D
        self.gOgpDp = np.zeros((self.bins,3,3)) # MgpDp
        self.gODpf = np.zeros((self.bins,3,3)) # MDpf
        self.gODff = np.zeros((self.bins,3,3,3)) #MDff
        self.Dtrace = np.zeros(self.bins)
        self.binfreqs = np.zeros(self.bins)
        self.m_sum = np.zeros(self.bins)
        self.jump_sum = np.zeros(self.bins)
        self.jumpsq_sum = np.zeros(self.bins)
    def mu(self,x):
        return 6*power(x/2,-7)

    def discreteLaplacian(self):
        L = np.eye(self.bins)*-2.0/self.dy**2
        L[np.arange(1,self.bins-1),np.arange(1,self.bins-1)-1] = 1/self.dy**2
        L[np.arange(1,self.bins-1),np.arange(1,self.bins-1)+1] = 1/self.dy**2
        L[0,0]=L[0,2] = L[self.bins-1,self.bins-1]=L[self.bins-1,self.bins-3] = 1.0/self.dy**2
        L[0,1]=L[self.bins-1,self.bins-2]= -2.0/self.dy**2
        return L

    def discreteDerivative(self):
        r = np.zeros(self.bins)
        c = np.zeros(self.bins)
        r[1] = 0.5/self.dy
        r[self.bins-1]=-0.5/self.dy
        c[1]=-0.5/self.dy
        c[self.bins-1]=0.5/self.dy
        return toeplitz(r,c).T

    def residual(self,functionvals,Mf,Mg,Mgp):

        # loop through the reshaped observations
        # functionvals should be of size 2*bins
        # [f,g]

        if self.mode=='exact':
            # in this case, the function values are at the exact positions
            numobs = len(self.positions)
            f = functionvals[:numobs]
            g = functionvals[:numobs]
            fresidual = np.zeros(numobs)
            gresidual = np.zeros(numobs)
            for j in range(len(self.positions)):
                fresidual[j] = f[j]-0.5*(self.jumps[j]-(f[j]+self.m[j])*self.dt)
            return np.hstack((fresidual,gresidual))

        f = functionvals[:self.bins]
        g = functionvals[self.bins:]

        gp = np.gradient(g,self.dy)
        D = self.D0*np.exp(g)
        Dinv = np.exp(-g)/self.D0
        Dp = np.gradient(D,self.dy)

        tempfOD = np.zeros(self.bins)
        tempfOfD = np.zeros(self.bins)
        tempfODp = np.zeros(self.bins)

        tempgOD1 = np.zeros(self.bins)
        tempgOD2 = np.zeros(self.bins)
        tempgODp1 = np.zeros(self.bins)
        tempgODp2 = np.zeros(self.bins)
        tempgODf1 = np.zeros(self.bins)
        tempgODf2 = np.zeros(self.bins)
        tempgODinv = np.zeros(self.bins)
        tempgOgpDp = np.zeros(self.bins)
        tempgODpf = np.zeros(self.bins)
        tempgODff = np.zeros(self.bins)

        for k in range(self.bins):
            if k==0:
                Dvals = (D[0],D[0],D[1])
                fvals = (f[0],f[0],f[1])
                Dpvals = (Dp[0],Dp[0],Dp[1])
                gpvals = (gp[0],gp[0],gp[1])
                Dinvvals = (Dinv[0],Dinv[0],Dinv[1])
            elif k == (self.bins-1):
                Dvals = (D[self.bins-2],D[self.bins-1],D[self.bins-1])
                fvals = (f[self.bins-2],D[self.bins-1],D[self.bins-1])
                Dpvals = (Dp[self.bins-2],Dp[self.bins-1],Dp[self.bins-1])
                gpvals = (gp[self.bins-2],gp[self.bins-1],gp[self.bins-1])
                Dinvvals = (Dinv[self.bins-2],Dinv[self.bins-1],Dinv[self.bins-1])
            else:
                Dvals = D[k-1:k+2]
                fvals = f[k-1:k+2]
                Dpvals = Dp[k-1:k+2]
                gpvals = gp[k-1:k+2]
                Dinvvals = Dinv[k-1:k+2]
            tempfOD[k] = self.fOD[k].dot(Dvals)
            tempfODp[k] = self.fODp[k].dot(Dpvals)
            tempfOfD[k] = self.fOfD[k].dot(fvals).dot(Dvals)
            if self.mode == 'fonly': continue
            tempgOD1[k] = self.gOD1[k].dot(Dvals)
            tempgOD2[k] = self.gOD2[k].dot(Dvals)
            tempgODp1[k] = self.gODp1[k].dot(Dpvals)
            tempgODp2[k] = self.gODp2[k].dot(Dpvals)
            tempgODf1[k] = self.gODf1[k].dot(fvals).dot(Dvals)
            tempgODf2[k] = self.gODf2[k].dot(fvals).dot(Dvals)
            tempgODinv[k] = self.gODinv[k].dot(Dinvvals)
            tempgOgpDp[k] = self.gOgpDp[k].dot(Dpvals).dot(gpvals)
            tempgODpf[k] = self.gODpf[k].dot(Dpvals).dot(fvals)
            tempgODff[k] = self.gODff[k].dot(Dvals).dot(fvals).dot(fvals)
        fresidual = f+0.5*Mf.dot(self.fO1+tempfOD+tempfODp+tempfOfD)

        # need to fix the first and last bins

        if self.mode =='fonly':
            gresidual = np.zeros(self.bins)
            return np.hstack((fresidual,gresidual))
        gresidual = g+0.5*Mg.dot(self.gO11+tempgOD1+tempgODp1+tempgODf1
            +tempgODinv+tempgOgpDp+tempgODpf+tempgODff)+0.5*Mgp.dot(self.gO12 + tempgOD2 + tempgODp2
            +tempgODf2)

        # also compute the Jacobian

        # Jfres_f
        #J1 = eye(self.bins) +

        return np.hstack((fresidual,gresidual))

    # only return the residual of g
    def residual2(self,g,f,Mf,Mg,Mgp):

        # loop through the reshaped observations
        # functionvals should be of size 2*bins
        # [f,g]
        gp = np.gradient(g,self.dy)
        D = self.D0*np.exp(g)
        Dinv = np.exp(-g)/self.D0
        Dp = np.gradient(D,self.dy)

        tempfOD = np.zeros(self.bins)
        tempfOfD = np.zeros(self.bins)
        tempfODp = np.zeros(self.bins)

        tempgOD1 = np.zeros(self.bins)
        tempgOD2 = np.zeros(self.bins)
        tempgODp1 = np.zeros(self.bins)
        tempgODp2 = np.zeros(self.bins)
        tempgODf1 = np.zeros(self.bins)
        tempgODf2 = np.zeros(self.bins)
        tempgODinv = np.zeros(self.bins)
        tempgOgpDp = np.zeros(self.bins)
        tempgODpf = np.zeros(self.bins)
        tempgODff = np.zeros(self.bins)

        for k in range(self.bins):
            if k==0:
                Dvals = (D[0],D[0],D[1])
                fvals = (f[0],f[0],f[1])
                Dpvals = (Dp[0],Dp[0],Dp[1])
                gpvals = (gp[0],gp[0],gp[1])
                Dinvvals = (Dinv[0],Dinv[0],Dinv[1])
            elif k == (self.bins-1):
                Dvals = (D[self.bins-2],D[self.bins-1],D[self.bins-1])
                fvals = (f[self.bins-2],D[self.bins-1],D[self.bins-1])
                Dpvals = (Dp[self.bins-2],Dp[self.bins-1],Dp[self.bins-1])
                gpvals = (gp[self.bins-2],gp[self.bins-1],gp[self.bins-1])
                Dinvvals = (Dinv[self.bins-2],Dinv[self.bins-1],Dinv[self.bins-1])
            else:
                Dvals = D[k-1:k+2]
                fvals = f[k-1:k+2]
                Dpvals = Dp[k-1:k+2]
                gpvals = gp[k-1:k+2]
                Dinvvals = Dinv[k-1:k+2]
            tempfOD[k] = self.fOD[k].dot(Dvals)
            tempfODp[k] = self.fODp[k].dot(Dpvals)
            tempfOfD[k] = self.fOfD[k].dot(fvals).dot(Dvals)
            if self.mode == 'fonly': continue
            tempgOD1[k] = self.gOD1[k].dot(Dvals)
            tempgOD2[k] = self.gOD2[k].dot(Dvals)
            tempgODp1[k] = self.gODp1[k].dot(Dpvals)
            tempgODp2[k] = self.gODp2[k].dot(Dpvals)
            tempgODf1[k] = self.gODf1[k].dot(fvals).dot(Dvals)
            tempgODf2[k] = self.gODf2[k].dot(fvals).dot(Dvals)
            tempgODinv[k] = self.gODinv[k].dot(Dinvvals)
            tempgOgpDp[k] = self.gOgpDp[k].dot(Dpvals).dot(gpvals)
            tempgODpf[k] = self.gODpf[k].dot(Dpvals).dot(fvals)
            tempgODff[k] = self.gODff[k].dot(Dvals).dot(fvals).dot(fvals)
        fresidual = f+0.5*Mf.dot(self.fO1+tempfOD+tempfODp+tempfOfD)

        # need to fix the first and last bins

        if self.mode =='fonly':
            gresidual = np.zeros(self.bins)
            return np.hstack((fresidual,gresidual))
        gresidual = g+0.5*Mg.dot(self.gO11+tempgOD1+tempgODp1+tempgODf1
            +tempgODinv+tempgOgpDp+tempgODpf+tempgODff)+0.5*Mgp.dot(self.gO12 + tempgOD2 + tempgODp2
            +tempgODf2)

        # also compute the Jacobian

        # Jfres_f
        #J1 = eye(self.bins) +

        return gresidual

    def guessD0(self,precision):
        # MLE estimate the jumps
        N = np.int(2*power(precision,-2))
        print N
        tailjumps = self.jumps[-N:]
        tailpos = self.positions[-N:]
        tails = self.m[-N:]

        self.ND0 = N
        D0 = (np.sqrt(np.sum(tails**2)*np.sum(tailjumps**2)+N**2)-N)/(self.dt*np.sum(tails**2))
        print D0
        self.D0 = D0

    def computeHff(self,f,g,Rf):
        '''
            $\Sigma_{ff}$ matrix, approximated on the grid
        '''
        # set up self.Hff with corresponding matrix
        # use the woodbury matrix formula, utilize the existing prior matrices
        D = np.exp(g)*self.D0
        dk = np.tile(self.binfreqs*D,(self.bins,1))*self.dt/2.
        dk2=Rf*dk
        temp = solve(np.eye(self.bins)+dk2,Rfy)
        return 0.5*(temp+temp.T)

    def computeHgg(self,f,g,Rg,Rgy,Rgyz):
        # set up self.Hgg with correspond matrix
        # also should give other two derivatives
        D = np.exp(g)*self.D0
        gp = np.gradient(g,self.dy)
        jumpsq = np.zeros(self.bins)
        driftsq = np.zeros(self.bins)
        driftsum = np.zeros(self.bins)
        for k in np.arange(self.bins):
            jumpsq[k] = sum(self.jumps[self.binpos==k]**2)
            driftsq[k] = sum( (D[k]*(f[k]+self.m[self.binpos==k])+gp[k]*D[k])**2)*self.dt*self.dt
            driftsum[k] = sum( D[k]*(f[k]+self.m[self.binpos==k])+gp[k]*D[k])*self.dt
        a1 = (jumpsq + driftsq)/D/4.0/self.dt
        a2 = driftsum/2.
        a4 = D*self.dt/2.
        A1 = np.tile(a1,(self.bins,1))*Rg+np.tile(a2,(self.bins,1))*-Rgy
        A2 = np.tile(a2,(self.bins,1))*Rg+np.tile(a4,(self.bins,1))*-Rgy
        A3 = np.tile(a1,(self.bins,1))*Rgy+np.tile(a2,(self.bins,1))*Rgyz
        A4 = np.tile(a2,(self.bins,1))*Rgy+np.tile(a4,(self.bins,1))*Rgyz

        A5 = np.tile(a1,(self.bins,1))*Rgy+np.tile(a2,(self.bins,1))*Rgyz
        A6 = np.tile(a2,(self.bins,1))*Rgy+np.tile(a4,(self.bins,1))*Rgyz
        M = np.vstack((Rg,Rgy))
        A = np.vstack((np.hstack((A1,A2)) ,np.hstack((A3,A4)) ))
        Lambda = solve(np.eye(self.bins*2)+A.astype(np.float128),M)
        Hggy = Lambda[self.bins:,:]
        Hgg = Lambda[:self.bins,:]

        Hggyz = solve(np.eye(self.bins)+A6,Rgyz-A5.dot(Hggy.T))
        return 0.5*(Hgg+Hgg.T),Hggy,0.5*(Hggyz+Hggyz.T) # first and third should be symmetric. get rid of small errors

    def computeSigmaff(self,f,g,Rf,Hgg,Hggp,Hggpp):
        '''
            Takes f, g, gg Hesian inverse and first two derivatives
        '''
        crossvals = np.zeros(self.bins)
        D = self.D0*np.exp(g)
        gp = np.gradient(g,self.dy)
        # LOOP to compute cross term
        Dsum = self.binfreqs*D
        driftsum = np.zeros(self.bins)

        for k in np.arange(self.bins):
            driftsum[k] = sum( D[k]*(f[k]+self.m[self.binpos==k])+gp[k]*D[k])*self.dt
            pass
        D = np.exp(g)*self.D0
        dk = Rf*np.tile(Dsum,(self.bins,1))*self.dt/2.

        dcross1 = -0.25*(Rf* np.tile(driftsum,(self.bins,1))).dot(Hgg*np.tile(driftsum,(self.bins,1)).T)
        dcross2 = -0.25*(Rf* np.tile(Dsum*self.dt,(self.bins,1))).dot(Hggp*np.tile(driftsum,(self.bins,1)).T)
        dcross3 =-0.25*(Rf* np.tile(driftsum,(self.bins,1))).dot(Hggp.T*np.tile(driftsum,(self.bins,1)).T)
        dcross4 = -0.25*(Rf* np.tile(Dsum*self.dt,(self.bins,1))).dot(Hggpp.T*np.tile(Dsum*self.dt,(self.bins,1)).T)
        temp = solve(np.eye(self.bins)+dk + dcross1+dcross2+dcross3+dcross4,Rf)
        return 0.5*(temp+temp.T)

    def computeSigmagg(self,f,g,Hff,Rg,Rgy,Rgyz):
        D = np.exp(g)*self.D0
        gp = np.gradient(g,self.dy)
        jumpsq = np.zeros(self.bins)
        driftsq = np.zeros(self.bins)
        Dsum = self.binfreqs*D
        driftsum = np.zeros(self.bins)
        for k in np.arange(self.bins):
            jumpsq[k] = sum(self.jumps[self.binpos==k]**2)
            driftsq[k] = sum( (D[k]*(f[k]+self.m[self.binpos==k])+gp[k]*D[k])**2)*self.dt*self.dt
            driftsum[k] = sum( D[k]*(f[k]+self.m[self.binpos==k])+gp[k]*D[k])*self.dt

        cross1 = -0.25*(Rg* np.tile(driftsum,(self.bins,1))).dot(Hff*np.tile(driftsum,(self.bins,1)).T)
        cross2 = 0.25*(Rgy*np.tile(Dsum,(self.bins,1))).dot(Hff*np.tile(driftsum,(self.bins,1)).T)
        cross3 = -0.25*(Rg* np.tile(driftsum,(self.bins,1))).dot(Hff*np.tile(Dsum,(self.bins,1)).T)
        cross4 = 0.25*(Rgy*np.tile(Dsum,(self.bins,1))).dot(Hff*np.tile(Dsum,(self.bins,1)).T)

        a1 = (jumpsq + driftsq)/D[k]/4.0/self.dt
        a2 = driftsum/2.
        a4 = D*self.dt/2.
        A1 = np.tile(a1,(self.bins,1))*Rg+np.tile(a2,(self.bins,1))*Rgy.T+cross1+cross2
        A2 = np.tile(a2,(self.bins,1))*Rg+np.tile(a4,(self.bins,1))*Rgy.T+cross3+cross4
        A3 = np.tile(a1,(self.bins,1))*Rgy+np.tile(a2,(self.bins,1))*Rgyz
        A4 = np.tile(a2,(self.bins,1))*Rgy+np.tile(a4,(self.bins,1))*Rgyz
        M = np.vstack((Rg,Rgy))
        A = np.vstack((np.hstack((A1,A2)) ,np.hstack((A3,A4)) ))
        Lambda1 = solve(np.eye(self.bins*2)+A,M)
        temp = Lambda1[:self.bins,:]
        return 0.5*(temp+temp.T)


    def processtrajectories(self,n=None):
        # processes next n trajectories
        if n == None:
            n = self.numtrajectories-self.trajectoryindex

        current = self.trajectoryindex
        # do some pre-processing for use in approximate inference
        # this pre-processing should be independent of the kernel chosen,
        # so that this time-intensive step need-not be repeated for
        # different choice of regularization
        # do sanity check to make sure that n is not too large

        if(current+n > self.numtrajectories):
            n = self.numtrajectories - current
            print "Inputed number of trajectories to process is too high, using ", n, " trajectories instead."

        # current starts at 0
        trajectories = self.trajectories[:,current:current+n]
        positions = np.ndarray.flatten( (trajectories[:-1,:]).transpose())
        indices = np.argsort(positions)
        times = np.tile(self.times[:-1],n)[indices]
        jumps = np.ndarray.flatten((trajectories[1:,:] - trajectories[:-1,:]).transpose())
        jumps = jumps[indices]
        positions = positions[indices]
        binpos = np.floor((positions-self.x0)/self.dy)
        muval = self.mu(positions)
        spring = self.x0+times*self.V # location of the spring
        m = muval - self.K*(positions-spring)

        # compute the coefficient matrices
        #print("Initializing coefficient matrices")

        numobs = len(positions)
        currentbin = 0
        freq = 0
        dist = positions - (self.midpoints[0]+self.dy*binpos)
        a = 0.5*power(dist/self.dy,2)-0.5*dist/self.dy
        b = 1.0-dist*dist/self.dy/self.dy
        c = 0.5*power(dist/self.dy,2)+0.5*dist/self.dy
        aa = a*a
        ab = a*b
        ac = a*c
        bb = b*b
        bc = b*c
        cc = c*c
        prog = ProgressBar(self.bins)
        #print("Processing the data, this can take a while...")

        jumpsum, groups = self.sum_by_group(jumps,binpos)
        jumpsqsum, groups = self.sum_by_group(jumps**2,binpos)
        msum, groups = self.sum_by_group(m,binpos)
        asum, groups = self.sum_by_group(a,binpos)
        bsum, groups = self.sum_by_group(b,binpos)
        csum, groups = self.sum_by_group(c,binpos)
        masum, groups = self.sum_by_group(m*a,binpos)
        mbsum, groups = self.sum_by_group(m*b,binpos)
        mcsum, groups = self.sum_by_group(m*c,binpos)
        aasum, groups = self.sum_by_group(aa,binpos)
        bbsum, groups = self.sum_by_group(bb,binpos)
        ccsum, groups = self.sum_by_group(cc,binpos)
        absum, groups = self.sum_by_group(ab,binpos)
        acsum, groups = self.sum_by_group(ac,binpos)
        bcsum, groups = self.sum_by_group(bc,binpos)
        bincounts, groups = self.sum_by_group(np.ones(binpos.shape[0]),binpos)
        ajumpsum, groups = self.sum_by_group(a*jumps,binpos)
        bjumpsum, groups = self.sum_by_group(b*jumps,binpos)
        cjumpsum, groups = self.sum_by_group(c*jumps,binpos)

        jump_sum = np.zeros(self.bins)
        jumpsq_sum = np.zeros(self.bins)
        m_sum = np.zeros(self.bins)
        a_sum = np.zeros(self.bins)
        b_sum = np.zeros(self.bins)
        c_sum = np.zeros(self.bins)
        ma_sum = np.zeros(self.bins)
        mb_sum = np.zeros(self.bins)
        mc_sum = np.zeros(self.bins)
        aa_sum = np.zeros(self.bins)
        bb_sum = np.zeros(self.bins)
        cc_sum = np.zeros(self.bins)
        ab_sum = np.zeros(self.bins)
        bc_sum = np.zeros(self.bins)
        ac_sum = np.zeros(self.bins)


        for k in range(self.bins):
            try:
                jump_sum[k] = jumpsum[groups==k]
                jumpsq_sum[k] = jumpsqsum[groups==k]
                m_sum[k] = msum[groups==k]
                a_sum[k] = asum[groups==k]
                b_sum[k] = bsum[groups==k]
                c_sum[k] = csum[groups==k]
                ma_sum[k] = masum[groups==k]
                mb_sum[k] = mbsum[groups==k]
                mc_sum[k] = mcsum[groups==k]
                aa_sum[k] = aasum[groups==k]
                bb_sum[k] = bbsum[groups==k]
                cc_sum[k] = ccsum[groups==k]
                ab_sum[k] = absum[groups==k]
                ac_sum[k] = acsum[groups==k]
                bc_sum[k] = bcsum[groups==k]
                self.binfreqs[k] += bincounts[groups==k]

            except:
                continue


        self.fO1 += -jump_sum
        self.fODp[:,0] += a_sum*self.dt
        self.fODp[:,1] += b_sum*self.dt
        self.fODp[:,2] += c_sum*self.dt
        self.fOD[:,0] += ma_sum*self.dt
        self.fOD[:,1] += mb_sum*self.dt
        self.fOD[:,2] += mc_sum*self.dt
        self.fOfD[:,0,0] += aa_sum*self.dt
        self.fOfD[:,1,1] += bb_sum*self.dt
        self.fOfD[:,2,2] += cc_sum*self.dt
        self.fOfD[:,0,1] += ab_sum*self.dt
        self.fOfD[:,1,0] += ab_sum*self.dt # redundant
        self.fOfD[:,0,2] += ac_sum*self.dt
        self.fOfD[:,2,0] += ac_sum*self.dt
        self.fOfD[:,1,2] += bc_sum*self.dt
        self.fOfD[:,2,1] += bc_sum*self.dt

        for k in range(self.bins):

            prog.animate(k+1)
            if self.mode != 'fonly':
                self.gODinv[k,0] += -np.sum(jumps[binpos==k]**2*a[binpos==k])*0.5/self.dt
                self.gODinv[k,1] += -np.sum(jumps[binpos==k]**2*b[binpos==k])*0.5/self.dt
                self.gODinv[k,2] += -np.sum(jumps[binpos==k]**2*c[binpos==k])*0.5/self.dt
                self.gODf1[k,0,0] += np.sum(aa[binpos==k]*m[binpos==k])*self.dt*0.5
                self.gODf1[k,1,1] += np.sum(bb[binpos==k]*m[binpos==k])*self.dt*0.5
                self.gODf1[k,2,2] += np.sum(cc[binpos==k]*m[binpos==k])*self.dt*0.5
                gODf1ab = np.sum(ab[binpos==k]*m[binpos==k])*self.dt*0.5
                gODf1bc = np.sum(bc[binpos==k]*m[binpos==k])*self.dt*0.5
                gODf1ac = np.sum(ac[binpos==k]*m[binpos==k])*self.dt*0.5
                self.gODf1[k,0,1] += gODf1ab
                self.gODf1[k,1,0] += gODf1ab
                self.gODf1[k,0,2] += gODf1ac
                self.gODf1[k,2,0] += gODf1ac
                self.gODf1[k,1,2] += gODf1bc
                self.gODf1[k,2,1] += gODf1bc

                # aaa, aab, aac
                self.gODff[k,0,0,0] += np.sum(a[binpos==k]**3)*self.dt*0.5
                self.gODff[k,1,1,1] += np.sum(b[binpos==k]**3)*self.dt*0.5
                self.gODff[k,2,2,2] += np.sum(c[binpos==k]**3)*self.dt*0.5
                # aab terms
                gODfaab = np.sum(a[binpos==k]**2*b[binpos==k])*self.dt*0.5
                self.gODff[k,0,0,1] += gODfaab
                self.gODff[k,0,1,0] += gODfaab
                self.gODff[k,1,0,0] += gODfaab
                # aac terms
                gODfaac = np.sum(a[binpos==k]**2*c[binpos==k])*self.dt*0.5
                self.gODff[k,0,0,2] += gODfaac
                self.gODff[k,0,2,0] += gODfaac
                self.gODff[k,2,0,0] += gODfaac
                #abb terms
                gODfabb = np.sum(a[binpos==k]*b[binpos==k]**2)*self.dt*0.5
                self.gODff[k,0,1,1] += gODfaab
                self.gODff[k,1,0,1] += gODfaab
                self.gODff[k,1,1,0] += gODfaab
                #bbc terms
                gODfbbc = np.sum(b[binpos==k]**2*c[binpos==k])*self.dt*0.5
                self.gODff[k,1,1,2] += gODfbbc
                self.gODff[k,1,2,1] += gODfbbc
                self.gODff[k,2,1,1] += gODfbbc
                #acc terms
                gODfacc = np.sum(a[binpos==k]*c[binpos==k]**2)*self.dt*0.5
                self.gODff[k,0,2,2] += gODfacc
                self.gODff[k,2,0,2] += gODfacc
                self.gODff[k,2,2,0] += gODfacc
                #bcc terms
                gODfbcc = np.sum(b[binpos==k]*c[binpos==k]**2)*self.dt*0.5
                self.gODff[k,1,2,2] += gODfbcc
                self.gODff[k,2,1,2] += gODfbcc
                self.gODff[k,2,2,1] += gODfbcc
                # abc terms
                gODfabc = np.sum(a[binpos==k]*b[binpos==k]*c[binpos==k])*self.dt*0.5
                self.gODff[k,0,1,2] += gODfabc
                self.gODff[k,1,2,0] += gODfabc
                self.gODff[k,2,0,1] += gODfabc
                self.gODff[k,2,1,0] += gODfabc
                self.gODff[k,1,0,2] += gODfabc
                self.gODff[k,0,2,1] += gODfabc

        if self.mode != "fonly":
            self.gO11 = self.binfreqs
            self.gOD1 = self.fOD*0.5
            self.gODf2 = self.fOfD
            self.gOgpDp = self.gODf2*0.5
            self.gOD2 = self.fOD
            self.gODp1 = self.fOD
            self.gODpf = self.fOfD

            self.gODp2 = self.fODp
            self.gO12 = self.fO1

        self.trajectoryindex += n

        # merge data with prior data if available

        try:
            newpositions = np.append(self.positions,positions)
            indices = np.argsort(newpositions)
            self.positions = newpositions[indices]
            self.jumps = np.append(self.jumps,jumps)[indices]
            self.a = np.append(self.a,a)[indices]
            self.b = np.append(self.b,b)[indices]
            self.c = np.append(self.c,c)[indices]
            self.binpos = np.append(self.binpos,binpos)[indices]
            self.m = np.append(self.m,m)[indices]
            self.sortedtimes = np.append(self.sortedtimes,times)[indices]

        except:
            self.positions = positions
            self.jumps = jumps
            self.a = a
            self.b = b
            self.c = c
            self.binpos = binpos
            self.m = m
            self.sortedtimes = times

        try:
            self.jump_sum += jump_sum
            self.jumpsq_sum += jumpsq_sum
            self.m_sum += m_sum
        except:
            self.jump_sum = jump_sum
            self.jumpsq_sum = jumpsq_sum
            self.m_sum = m_sum

        return

    def readmatfile(self, filename):
        matfile = loadmat(filename)
        self.trajectories = matfile['A'][:,1:]
        self.times = matfile['A'][:,0]
        self.T0 = self.times[0]
        self.Tf = self.times[-1]
        self.dt = self.times[1]-self.times[0]
        self.numtrajectories = self.trajectories.shape[1]
        self.trajectoryindex = 0
        #self.processtrajectories()

    #@void()
    def readtextfile(self, filename):
        A = np.loadtxt(filename)
        self.trajectories = A[:,1:]
        self.times = A[:,0]
        self.T0 =self.times[0]
        self.Tf = self.times[-1]
        self.dt = self.times[1]-self.times[0]
        self.numtrajectories = self.trajectories.shape[1]
        #self.processtrajectories()
        return

    def loadibwfiles(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".ibw"):
                    data = loadibw(path)
                    notes=dict(item.split(":",1) for item in data['wave']['note'].split('\r')[:-1])
                    K=float(notes['SpringConstant'])
                    time = data['wave']['wData'][:,3]
                    Defl = data['wave']['wData'][:,1]
                    LVDT = data['wave']['wData'][:,2]

        return

    def matrixR(self,beta,gamma):
        GM = np.fromfunction(lambda i,j: self.R(self.midpoints[0]+i*self.dy,self.midpoints[0]+j*self.dy,beta,gamma),(self.bins,self.bins))
        return GM

    def matrixRy(self,beta,gamma):
        return np.fromfunction(lambda i,j: self.Ry(self.midpoints[0]+i*self.dy,self.midpoints[0]+j*self.dy,beta,gamma),(self.bins,self.bins))

    def matrixRz(self,beta,gamma):
        return  np.fromfunction(lambda i,j: self.Rz(self.midpoints[0]+i*self.dy,self.midpoints[0]+j*self.dy,beta,gamma),(self.bins,self.bins))

    def matrixRyy(self,beta,gamma):
        return np.fromfunction(lambda i,j: self.Ryy(self.midpoints[0]+i*self.dy,self.midpoints[0]+j*self.dy,beta,gamma),(self.bins,self.bins))

    def matrixRyz(self,beta,gamma):
        return -self.matrixRyy(beta,gamma)

    def matrixRzz(self,beta,gamma):
        return self.matrixRyy(beta,gamma)

    def matrixRyyz(self,beta,gamma):
        GM = np.zeros((self.bins,self.bins))
        Gval = np.zeros(self.bins)
        for j in np.arange(self.bins):
            Gval[j] = self.R2(abs(j*self.dy),beta,gamma)
        for j in np.arange(self.bins):
            for k in np.arange(self.bins):
                GM[j,k]=Gval[abs(j-k)]*(-(abs(j-k)*self.dy)**2+3*gamma)/beta/gamma**-3*self.dy*(k-j)
        return GM

    def Hstar(self,f,g,theta):
        # evaluate the Hamiltonian at saddle point
        D = self.D0*np.exp(g)
        gp = np.gradient(g,self.dy)
        driftsum = np.zeros(self.bins)
        jumpsum = np.zeros(self.bins)
        jumpsqsum = np.zeros(self.bins)
        driftsqsum = np.zeros(self.bins)
        errsum = np.zeros(self.bins)
        for k in range(self.bins):
            driftsum[k]=sum( D[k]*(f[k]+self.m[self.binpos==k])+gp[k]*D[k])*self.dt
            jumpsum[k]=sum(self.jumps[self.binpos==k])
            jumpsqsum[k] = sum(self.jumps**2[self.binpos==k])
            driftsqsum[k] = sum( (D[k]*(f[k]+self.m[self.binpos==k])+gp[k]*D[k])**2)*self.dt*self.dt
            errsum[k] = sum( (self.jumps- ( D[k]*(f[k]+self.m[self.binpos==k])+gp[k]*D[k])*self.dt)**2/4./D/self.dt)

        fnorm = sum(f*(jumpsum-driftsum))/2.
        gnorm = sum(gp*(jumpsum-driftsum))/2. - sum(g*(1-(jumpsqsum-driftsqsum)/(2*D*self.dt)  ))
        Dtrace = 0.5*sum(np.log(D)*self.binfreqs)
        return 0.5*fnorm+0.5*gnorm+Dtrace+sum(errsum)

    def logmarginal(self,f,g,theta):
        Hstar = self.Hstar(f,g,theta)
        # compute the two trace log terms

    def solveDiscrete(self,betaf,betag):
        solver = scipy.optimize.root(self.residual,np.zeros(self.bins*2),(np.eye(self.bins)/betaf,np.eye(self.bins)/betag,np.eye(self.bins)/betag),method='lm')
        solution = {
            'optimout': solver,
            'fstar': solver.x[:self.bins],
            'x': self.midpoints,
            'gstar': solver.x[self.bins:],
            'Dstar': self.D0*np.exp(solver.x[self.bins:])
        }
        return solution

    def solveSemiclassically(self,params,err=False):
        # solve the inverse problem and return all of the relevant things
        # like the solution, covariance matrices, and other matrices
        betaf = params['betaf']
        gammaf = params['gammaf']
        betag = params['betag']
        gammag = params['gammag']
        Rf = self.matrixR(betaf,gammaf)
        Rg = self.matrixR(betag,gammag)
        Rgy = self.matrixRy(betag,gammag)
        Rgz = Rgy.T
        Rgyz = self.matrixRyz(betag,gammag)
        #print("Reconstructing f and g...")
        optimout = scipy.optimize.root(self.residual,np.zeros(self.bins*2),(Rf,Rg,Rgz), tol=10e-7)
        '''
        if not optimout.success:
            print "'hybr' optimization method not sucessful, trying the krylov method"
            optimout = scipy.optimize.root(self.residual,np.zeros(self.bins*2),(Rf,Rg,Rgz),method="krylov",tol=10e-7)
        if not optimout.success:
            print "'krylov' not sucessful, trying 'lm'"
            optimout = scipy.optimize.root(self.residual,np.zeros(self.bins*2),(Rf,Rg,Rgz),method="lm", tol=10e-7)
        '''
        if optimout.success:
            f = optimout.x[:self.bins]
            g = optimout.x[self.bins:]
            D = self.D0*np.exp(g)
        else:
            print "Optimization was not successful for some reason"
            solution = {
                'x': self.midpoints,
                'Rf': Rf,
                'Rg': Rg,
                'fstar': np.zeros(self.bins),
                'gstar': np.zeros(self.bins),
                'Dstat': np.zeros(self.bins),
                'params': params,
                'optimout': optimout,
                'marlike': 1e300,
            }
            return solution
        # compute Hff
        dk = np.tile(self.binfreqs*D,(self.bins,1))*self.dt/2.
        dk2=Rf*dk
        gp = np.gradient(g,self.dy)
        # compute Hgg
        # A = D(f+m+g')dt
        # A2 = D^2(f+m+g')^2 = D^2(f^2+m^2+g'^2+2fm + 2mg' + 2fg')dtdt
        #driftsum, groups = self.sum_by_group(drift,self.binpos)
        #driftsqsum, groups = self.sum_by_group(drift*drift,self.binpos)
        drift_sum = D*(f*self.binfreqs+self.m_sum+gp*self.binfreqs)*self.dt
        m2sum, groups = self.sum_by_group(self.m**2,self.binpos)
        mjumpsum, groups = self.sum_by_group(self.m*self.jumps,self.binpos)
        m2_sum = np.zeros(self.bins)
        mjump_sum = np.zeros(self.bins)
        for k in np.arange(self.bins):
            try:
                m2_sum[k] = m2sum[groups==k]
            except:
                m2_sum[k] = 0
            try:
                mjump_sum[k] = mjumpsum[groups==k]
            except:
                mjump_sum[k] = 0
        driftsq_sum = D**2*(f**2*self.binfreqs+m2_sum+gp**2*self.binfreqs \
            +2*f*self.m_sum+2*self.m_sum*gp+2*f*gp*self.binfreqs)*self.dt**2
        err_sum = self.jumpsq_sum/4.0/D/self.dt + driftsq_sum/4.0/D/self.dt \
            - self.jump_sum*(f+gp)/2.0 - mjump_sum/2.0
        a1 = (self.jumpsq_sum + driftsq_sum)/D/4.0/self.dt
        a2 = drift_sum/2.
        a4 = D*self.dt/2.
        if(err):
            #print("Computing posterior covariance matrix")
            temp = solve(np.eye(self.bins)+dk2,Rf)
            Hff = 0.5*(temp+temp.T)
            A1 = np.tile(a1,(self.bins,1))*Rg+np.tile(a2,(self.bins,1))*Rgz
            A2 = np.tile(a2,(self.bins,1))*Rg+np.tile(a4,(self.bins,1))*Rgz
            A3 = np.tile(a1,(self.bins,1))*Rgy+np.tile(a2,(self.bins,1))*Rgyz
            A4 = np.tile(a2,(self.bins,1))*Rgy+np.tile(a4,(self.bins,1))*Rgyz

            A5 = np.tile(a1,(self.bins,1))*Rgy+np.tile(a2,(self.bins,1))*Rgyz
            A6 = np.tile(a2,(self.bins,1))*Rgy+np.tile(a4,(self.bins,1))*Rgyz
            M = np.vstack((Rg,Rgy))
            A = np.vstack((np.hstack((A1,A2)) ,np.hstack((A3,A4)) ))
            Lambda = solve(np.eye(self.bins*2)+A.astype(np.float128),M)
            Hggy = Lambda[self.bins:,:]
            Hgg = Lambda[:self.bins,:]
            Hggyz = solve(np.eye(self.bins)+A6,Rgyz-A5.dot(Hggy.T))

            # compute \Sigma_{ff}

            Dsum = self.binfreqs*D

            dcross1 = -0.25*(Rf* np.tile(drift_sum,(self.bins,1))).dot(Hgg*np.tile(drift_sum,(self.bins,1)).T)
            dcross2 = -0.25*(Rf* np.tile(Dsum*self.dt,(self.bins,1))).dot(Hggy*np.tile(drift_sum,(self.bins,1)).T)
            dcross3 =-0.25*(Rf* np.tile(drift_sum,(self.bins,1))).dot(Hggy.T*np.tile(drift_sum,(self.bins,1)).T)
            dcross4 = -0.25*(Rf* np.tile(Dsum*self.dt,(self.bins,1))).dot(Hggyz.T*np.tile(Dsum*self.dt,(self.bins,1)).T)
            temp = solve(np.eye(self.bins)+dk2 + dcross1+dcross2+dcross3+dcross4,Rf)
            Sigmaff = 0.5*(temp+temp.T)

            # compute \Sigma_{gg}
            cross1 = -0.25*(Rg* np.tile(drift_sum,(self.bins,1))).dot(Hff*np.tile(drift_sum,(self.bins,1)).T)
            cross2 = 0.25*(Rgy*np.tile(Dsum,(self.bins,1))).dot(Hff*np.tile(drift_sum,(self.bins,1)).T)
            cross3 = -0.25*(Rg* np.tile(drift_sum,(self.bins,1))).dot(Hff*np.tile(Dsum,(self.bins,1)).T)
            cross4 = 0.25*(Rgy*np.tile(Dsum,(self.bins,1))).dot(Hff*np.tile(Dsum,(self.bins,1)).T)

            A7 = np.tile(a1,(self.bins,1))*Rg+np.tile(a2,(self.bins,1))*Rgy.T+cross1+cross2
            A8 = np.tile(a2,(self.bins,1))*Rg+np.tile(a4,(self.bins,1))*Rgy.T+cross3+cross4
            M = np.vstack((Rg,Rgy))
            A = np.vstack((np.hstack((A7,A8)) ,np.hstack((A3,A4)) ))
            Lambda1 = solve(np.eye(self.bins*2)+A,M)
            temp = Lambda1[:self.bins,:]
            Sigmagg = 0.5*(temp+temp.T)
            Sigmaggy = Lambda1[self.bins:,:]

        #print("Computing marginal likelihood")

        # compute H[f*,g*]


        fnorm = np.sum(f*(self.jump_sum-drift_sum))*0.5
        gnorm = 0.5*np.sum(gp*(self.jump_sum-drift_sum)) - \
            0.5*np.sum(g*(self.binfreqs-(self.jumpsq_sum-driftsq_sum)/D/self.dt/2))

        # sanity check
        if fnorm<0 or gnorm<0:
            print("norm weirdness fnorm: %d, gnorm %d" % (fnorm,gnorm))
            solution = {
                'x': self.midpoints,
                'Rf': Rf,
                'Rg': Rg,
                'fstar': np.zeros(self.bins),
                'gstar': np.zeros(self.bins),
                'Dstar': np.zeros(self.bins),
                'params': params,
                'optimout': optimout,
                'marlike': 1e300,
            }
            return solution

        Dtrace = 0.5*np.sum(np.log(D)*self.binfreqs)
        Hstar = 0.5*fnorm+0.5*gnorm+Dtrace+sum(err_sum)

        # eigenvalues
        # frequency square matrix
        freqsq = np.outer(self.binfreqs,self.binfreqs)
        eigen1 = 1+np.real(eigvals(dk2))
        A9 =  np.tile(a1,(self.bins,1))*Rg+np.tile(a2,(self.bins,1))*Rgy
        A10 = np.tile(a2,(self.bins,1))*Rgz+np.tile(a4,(self.bins,1))*Rgyz

        eigen2 = np.real(eigvals(A9))
        eigen3 = np.real(eigvals(A10))
        marginallik = Hstar - np.sum(np.log(eigen1)) - np.sum(np.log(1+np.abs(eigen2)))-np.sum(np.log(1+np.abs(eigen3)))

        # do some sanity check


        solution = {
            'x': self.midpoints,
            'fstar': f,
            'gstar': g,
            'Dstar': D,
            'D0': self.D0,
            'fnorm': fnorm,
            'gnorm': gnorm,
            'err_sum': err_sum,
            'params': params,
            'optimout': optimout,
            'Hstar': Hstar,
            'feigen': eigen1,
            'geigen1': 1+eigen2,
            'geigen2': 1+eigen3,
            'marlike': marginallik,
            'driftsq_sum': driftsq_sum,
            'drift_sum': drift_sum
        }
        if(err):
            solution['Hgg'] = Hgg,
            solution['Hff'] = Hff,
            solution['Hggy'] = Hggy,
            solution['Hggyz'] = Hggyz,
            solution['Sigmaff'] = Sigmaff,
            solution['Sigmagg'] = Sigmagg
        print("I was able to obtain a solution with log marginal likelihood: %d" % (marginallik))
        return solution

    def R(self,x,y,beta,gamma): return np.exp(-(x-y)**2/2.0/gamma)*beta# - np.exp(-(x+y)**2/2.0/gamma)*beta
    def Ry(self,y,z,beta,gamma):
        return np.exp(-(y-z)**2/2.0/gamma)*beta*(z-y)/gamma# + np.exp(-(y+z)**2/2.0/gamma)*beta*(z+y)/gamma
    def Rz(self,y,z,beta,gamma):
        return -np.exp(-(y-z)**2/2.0/gamma)*beta*(z-y)/gamma #+ np.exp(-(y+z)**2/2.0/gamma)*beta*(z+y)/gamma
    def Ryy(self,x,y,beta,gamma): return np.exp(-(x-y)**2/2.0/gamma)*((x-y)**2-gamma)*beta/gamma/gamma
    def Ryyy(self,x,y,beta,gamma): return  np.exp(-(x-y)**2/2.0/gamma)*(-(x-y)**2+3*gamma)*(x-y)*beta/gamma**-3
    def sum_by_group(self,values, groups):
        order = np.argsort(groups)
        groups = groups[order]
        values = values[order]
        values.cumsum(out=values)
        index = np.ones(len(groups), 'bool')
        index[:-1] = groups[1:] != groups[:-1]
        values = values[index]
        groups = groups[index]
        values[1:] = values[1:] - values[:-1]
        return values, groups
