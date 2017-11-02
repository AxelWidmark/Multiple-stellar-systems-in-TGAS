#!/Users/axelwidmark/anaconda/bin/python
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import math
import mcmcaw as mc
from scipy.integrate import dblquad
import random
pi=math.pi




# TODO
# fix normalizations, now without covariance
# add selection function
# 
#

class model():
    
    # Initiation function
    # thinningfactor is for thinnning out the sample
    def __init__(self,thinningfactor=1):
        npzfile = np.load("./TGASx2MASS_cut_d<200pc.npz")
        parlist = npzfile["parlist"]
        self.parlist = parlist
        parerrlist = npzfile["parerrlist"]
        colorlist = npzfile["colorlist"]
        colorerrlist = npzfile["colorerrlist"]
        self.magJlims = np.array([3.,6.])
        self.JmKlims = np.array([0.5,0.8])
        self.length = len(colorlist)
        self.JmKlist = np.array([colorlist[i][0]-colorlist[i][2] for i in range(0,self.length,thinningfactor)])
        self.magJlist = np.array([colorlist[i][0]-5.*(np.log10(1e3/parlist[i])-1.) for i in range(0,self.length,thinningfactor)])
        self.cov = np.array([    [ [colorerrlist[i][0]**2.+colorerrlist[i][2]**2.,  colorerrlist[i][0]**2.],
                        [colorerrlist[i][0]**2.,  colorerrlist[i][0]**2.+(5.*parerrlist[i]/parlist[i]/np.log(10.))**2. ]   ]
                        for i in range(0,self.length,thinningfactor)])
        if thinningfactor!=1:
            self.length = len(self.magJlist)
        #plt.hist(parerrlist/parlist,100)
        #plt.xlabel('Parallax error / observed parallax')
        #plt.ylabel('Object count')
        #plt.show()
        #exit()
    
    # compute a bivariate gaussian value
    def bivariate(self,x,cov):
        invnorm = cov[0,0]*cov[1,1]-cov[0,1]**2.
        cov_inv = [[cov[1,1]/invnorm,-cov[0,1]/invnorm],[-cov[0,1]/invnorm,cov[0,0]/invnorm]]
        cov_det = cov[0,0]*cov[1,1]-cov[0,1]**2.
        return math.exp(-1./2.*np.dot(np.dot(cov_inv,x),x))/(2.*pi*math.sqrt(cov_det))
    
    # add two gaussian distributions for single stars together as a binary
    # this one operates in (c,M_J)-space
    # g1 and g2 should be on form [[mean(c),mean(M_J)], covariance matrix]
    def gaussian_sum(self,cM1,cov1,cM2,cov2):
        def add_magnitudes(Mvec):
            return -2.5*np.log10( sum([10.**(-Mi/2.5) for Mi in Mvec]) )
        def merge(cM1,cM2):
            M_k1 = cM1[1]-cM1[0]
            M_k2 = cM2[1]-cM2[0]
            M_j,M_k = [add_magnitudes([cM1[1],cM2[1]]),add_magnitudes([M_k1,M_k2])]
            return np.array([M_j-M_k,M_j])
        mean = merge(cM1,cM2)
        jac = lambda c_a,J_a,c_b,J_b:   np.matrix(  [[10.**(-(J_a-c_a)/2.5)/(10.**(-(J_a-c_a)/2.5)+10.**(-(J_b-c_b)/2.5)),
                    0.],
                [(10.**(-J_a/2.5)/(10.**(-J_a/2.5)+10.**(-J_b/2.5)))-(10.**(-(J_a-c_a)/2.5)/(10.**(-(J_a-c_a)/2.5)+10.**(-(J_b-c_b)/2.5))),
                    (10.**(-J_a/2.5)/(10.**(-J_a/2.5)+10.**(-J_b/2.5)))]]  )
        jacobian1 = jac(cM1[0],cM1[1],cM2[0],cM2[1])
        jacobian2 = jac(cM2[0],cM2[1],cM1[0],cM1[1])
        covA = np.dot(jacobian1,np.dot(cov1,np.transpose(jacobian1)))
        covB = np.dot(jacobian2,np.dot(cov2,np.transpose(jacobian2)))
        return mean,covA+covB
    
    # the straight line through (c,M_J)-space
    def muf(self,JmK,beta_1,beta_2):
        return beta_1+beta_2*(JmK-0.5)
    
    # generates gaussians of the singles distribution
    # returns cM,exp(-alpha*c),cov
    def single_gaussians(self,hyperparams,JmKlims=[0.4,.9]):
        alpha,beta_1,beta_2,gamma,binary_fraction,triple_fraction = hyperparams
        JmKs = np.linspace(JmKlims[0],JmKlims[1],int((JmKlims[1]-JmKlims[0])/0.05))
        def cov(JmK):
            JmKwidth = 0.025
            eig1 = np.array([1.,beta_2])
            eig2 = np.array([beta_2,-1.])
            eigs = np.transpose(np.array([eig1,eig2]))
            eigs_inv = np.linalg.inv(eigs)
            eigvals = np.matrix( [[4.*(beta_2**2.+1.)*JmKwidth**2.,0.],[0.,4.*gamma**2.]] )
            return np.dot(eigs,np.dot(eigvals,eigs_inv))
        return [ np.array([[JmK,self.muf(JmK,beta_1,beta_2)] for JmK in JmKs]), np.exp(-alpha*JmKs), np.array([cov(JmK) for JmK in JmKs]) ]
    
    # generates gaussians of the doubles distribution (by which I mean binaries)
    # does so from the single gaussians by adding them up with gaussian sum
    def double_gaussians(self,hyperparams):
        width_JmK = 1. # make hyperparam?
        alpha,beta_1,beta_2,gamma,binary_fraction,triple_fraction = hyperparams
        sg_cMs,sg_nums,sg_covs = self.single_gaussians(hyperparams,JmKlims=[0.5-width_JmK,.8+width_JmK])
        dg_cMs = []
        dg_nums = []
        dg_covs = []
        for i in range(0,len(sg_cMs),1):
            for j in range(i,len(sg_cMs),2):
                if abs(sg_cMs[i][0]-sg_cMs[j][0])<width_JmK and sg_cMs[j][0]>0.5 and sg_cMs[i][0]<0.8:
                    mean,cov = self.gaussian_sum(sg_cMs[i],sg_covs[i],sg_cMs[j],sg_covs[j])
                    if mean[0]>=0.4 and mean[0]<=0.9:
                        dg_cMs.append(mean)
                        dg_nums.append(math.exp(-alpha*mean[0]))
                        dg_covs.append(cov)
        return [ np.array(dg_cMs), np.array(dg_nums), np.array(dg_covs) ]
    
    # generates gaussians of the triples distribution
    def triple_gaussians(self,hyperparams):
        width_JmK = 1.
        alpha,beta_1,beta_2,gamma,binary_fraction,triple_fraction = hyperparams
        sg_cMs,sg_nums,sg_covs = self.single_gaussians(hyperparams,JmKlims=[0.5-width_JmK,.8+width_JmK])
        tg_cMs = []
        tg_nums = []
        tg_covs = []
        for i in range(0,len(sg_cMs),1):
            for j in range(i,len(sg_cMs),4):
                for k in range(j,len(sg_cMs),4):
                    if abs(sg_cMs[i][0]-sg_cMs[k][0])<width_JmK and sg_cMs[k][0]>0.5 and sg_cMs[i][0]<0.8:
                        mean,cov = self.gaussian_sum(sg_cMs[i],sg_covs[i],sg_cMs[j],sg_covs[j])
                        mean,cov = self.gaussian_sum(mean,cov,sg_cMs[k],sg_covs[k])
                        if mean[0]>=0.4 and mean[0]<=0.9:
                            tg_cMs.append(mean)
                            tg_nums.append(math.exp(-alpha*mean[0]))
                            tg_covs.append(cov)
        return [ np.array(tg_cMs), np.array(tg_nums), np.array(tg_covs) ]
    
    # generates data points from model
    def generate_objects(self,n,hyperparams,obj_type='single'):
        if obj_type=='single':
            cMs,nums,covs = self.single_gaussians(hyperparams)
        elif obj_type=='double':
            cMs,nums,covs = self.double_gaussians(hyperparams)
        elif obj_type=='triple':
            cMs,nums,covs = self.triple_gaussians(hyperparams)
        norm = sum(nums)
        objects = []
        for i in range(len(cMs)):
            num = int(nums[i]/norm*float(n)) + (np.random.rand()<(nums[i]/norm*float(n))%1.)
            objects_sub = np.random.multivariate_normal(cMs[i],covs[i],num)
            if objects==[]:
                objects = objects_sub
            else:
                objects = np.concatenate((objects,objects_sub),axis=0)
        print len(cMs)
        return np.transpose(objects)
    
    # the log posterior on population parameters
    def hyperposterior(self,hyperparams):
        alpha,beta_1,beta_2,gamma,binary_fraction,triples_fraction = hyperparams
        if binary_fraction<0. or binary_fraction>1. or triples_fraction<0. or triples_fraction>1. or gamma_1<0.:
            return -np.inf
        def priors(hyperparams):
            alpha,beta_1,beta_2,gamma_1,gamma_2,binary_fraction,triples_fraction = hyperparams
            return -1./2.*(binary_fraction/0.3)**2. -1./2.*(triples_fraction/0.1)**2. -1./2.*(gamma_2/10.)**2.-1./2.*(gamma_2/5.)**2.
        fraction = lambda cM,cov:1./4.*( math.erf((self.magJlims[1]-cM[1])/math.sqrt(2.*cov[1][1])) \
                                        -math.erf((self.magJlims[0]-cM[1])/math.sqrt(2.*cov[1][1])))* \
                                        ( math.erf((self.JmKlims[1]-cM[0])/math.sqrt(2.*cov[0][0])) \
                                        -math.erf((self.JmKlims[0]-cM[0])/math.sqrt(2.*cov[0][0])))
        sg_cMs,sg_nums,sg_covs = self.single_gaussians(hyperparams)
        norm_single = np.sum( [sg_nums[i]*fraction(sg_cMs[i],sg_covs[i]) for i in range(len(sg_cMs))] )
        dg_cMs,dg_nums,dg_covs = self.double_gaussians(hyperparams)
        norm_double = np.sum( [dg_nums[i]*fraction(dg_cMs[i],dg_covs[i]) for i in range(len(dg_cMs))] )
        tg_cMs,tg_nums,tg_covs = self.triple_gaussians(hyperparams)
        norm_triple = np.sum( [tg_nums[i]*fraction(tg_cMs[i],tg_covs[i]) for i in range(len(tg_cMs))] )
        gaussians = [[sg_cMs,sg_nums,sg_covs],[dg_cMs,dg_nums,dg_covs],[tg_cMs,tg_nums,tg_covs]]
        norms = np.array([norm_single,norm_double,norm_triple])
        fractions = np.array( [(1.-binary_fraction-triples_fraction),binary_fraction,triples_fraction] )
        res = priors(hyperparams)
        for i in range(self.length):
            JmK_i = self.JmKlist[i]
            magJ_i = self.magJlist[i]
            cov_i = self.cov[i]
            subres = 0.
            for k in range(3):
                for i_g in range(len(gaussians[k][0])):
                    mudiff = [JmK_i-gaussians[k][0][i_g][0],magJ_i-gaussians[k][0][i_g][1]]
                    if abs(mudiff[0])<0.2:
                        cov_sum = cov_i+gaussians[k][2][i_g]
                        subres += fractions[k]*gaussians[k][1][i_g]*self.bivariate(mudiff,cov_sum)/norms[k]
            res += np.log( subres )
        return res
    
    # posteriors on the objects in terms of what type
    # only necessary for plotting with function self.scatter
    def posteriors(self,hyperparams):
        alpha,beta_1,beta_2,gamma,binary_fraction,triples_fraction = hyperparams
        if binary_fraction<0. or binary_fraction>1. or triples_fraction<0. or triples_fraction>1. or gamma_1<0.:
            return -np.inf
        def priors(hyperparams):
            alpha,beta_1,beta_2,gamma_1,gamma_2,binary_fraction,triples_fraction = hyperparams
            return 0.#-1./2.*(binary_fraction/0.2)**2. -1./2.*(triples_fraction/0.05)**2. -1./2.*(gamma_2/10.)**2.
        fraction = lambda cM,cov:1./4.*( math.erf((self.magJlims[1]-cM[1])/math.sqrt(2.*cov[1][1])) \
                                        -math.erf((self.magJlims[0]-cM[1])/math.sqrt(2.*cov[1][1])))* \
                                        ( math.erf((self.JmKlims[1]-cM[0])/math.sqrt(2.*cov[0][0])) \
                                        -math.erf((self.JmKlims[0]-cM[0])/math.sqrt(2.*cov[0][0])))
        sg_cMs,sg_nums,sg_covs = self.single_gaussians(hyperparams)
        norm_single = np.sum( [sg_nums[i]*fraction(sg_cMs[i],sg_covs[i]) for i in range(len(sg_cMs))] )
        dg_cMs,dg_nums,dg_covs = self.double_gaussians(hyperparams)
        norm_double = np.sum( [dg_nums[i]*fraction(dg_cMs[i],dg_covs[i]) for i in range(len(dg_cMs))] )
        tg_cMs,tg_nums,tg_covs = self.triple_gaussians(hyperparams)
        norm_triple = np.sum( [tg_nums[i]*fraction(tg_cMs[i],tg_covs[i]) for i in range(len(tg_cMs))] )
        gaussians = [[sg_cMs,sg_nums,sg_covs],[dg_cMs,dg_nums,dg_covs],[tg_cMs,tg_nums,tg_covs]]
        norms = np.array([norm_single,norm_double,norm_triple])
        fractions = np.array( [(1.-binary_fraction-triples_fraction),binary_fraction,triples_fraction] )
        res = []
        for i in range(self.length):
            JmK_i = self.JmKlist[i]
            magJ_i = self.magJlist[i]
            cov_i = self.cov[i]
            subres = np.array( [0.,0.,0.] )
            for k in range(3):
                for i_g in range(len(gaussians[k][0])):
                    mudiff = [JmK_i-gaussians[k][0][i_g][0],magJ_i-gaussians[k][0][i_g][1]]
                    if abs(mudiff[0])<0.2:
                        cov_sum = cov_i+gaussians[k][2][i_g]
                        subres[k] += fractions[k]*gaussians[k][1][i_g]*self.bivariate(mudiff,cov_sum)/norms[k]
            res.append(subres)
        return np.array( res )
    
    # scatter plot with marginalized single/double/triple posteriors, etc.
    def scatter(self,hyperparams,hyperparamsvec=None):
        alpha,beta_1,beta_2,gamma,binary_fraction,triples_fraction = hyperparams
        if hyperparamsvec==None:
            single_posts = self.posteriors(hyperparams)
        else:
            single_posts = np.zeros([self.length,3])
            for hyp in hyperparamsvec:
                print hyp
                single_posts += np.array(self.posteriors(hyp))
        single_colors = [[sp[0]/sum(sp),0.6*sp[2]/sum(sp),sp[1]/sum(sp)] for sp in single_posts]
        single_dists = []
        double_dists = []
        triple_dists = []
        for i in range(self.length):
            if single_colors[i][0]>single_colors[i][1] and single_colors[i][0]>single_colors[i][2]:
                single_dists.append(1./self.parlist[i])
            elif single_colors[i][1]>single_colors[i][2]:
                double_dists.append(1./self.parlist[i])
            else:
                triple_dists.append(1./self.parlist[i])
        #plt.hist(single_dists,np.linspace(0.,0.2,21),histtype='step')
        #plt.hist(double_dists,np.linspace(0.,0.2,21),histtype='step')
        #plt.hist(triple_dists,np.linspace(0.,0.2,21),histtype='step')
        #plt.show()
        plt.subplot(1,3,1)
        plt.title('Data')
        from matplotlib import cm
        cmap = cm.get_cmap('RdBu')
        plt.scatter(self.JmKlist,self.magJlist,s=1,c=single_colors,edgecolors='none',vmin=0., vmax=1.)
        plt.xlim(self.JmKlims)
        plt.ylim(self.magJlims)
        plt.gca().invert_yaxis()
        plt.xlabel('$J-K$')
        plt.ylabel('$M_J$')
        plt.subplot(1,3,2)
        plt.title('Model')
        JmKsingles,magJsingles = self.generate_objects(8000*(1.-binary_fraction-triples_fraction),hyperparams)
        JmKdoubles,magJdoubles = self.generate_objects(8000*binary_fraction,hyperparams,obj_type='double')
        JmKtriples,magJtriples = self.generate_objects(8000*triples_fraction,hyperparams,obj_type='triple')
        plt.scatter(JmKsingles,magJsingles,s=1,edgecolors='none',c=[1.,0.,0.])
        plt.scatter(JmKdoubles,magJdoubles,s=1,edgecolors='none',c=[0.,0.,1.])
        plt.scatter(JmKtriples,magJtriples,s=1,edgecolors='none',c=[0.,0.6,0.])
        plt.xlim(self.JmKlims)
        plt.ylim(self.magJlims)
        plt.gca().invert_yaxis()
        plt.xlabel('$J-K$')
        #plt.ylabel('$M_J$')
        plt.subplot(1,3,3)
        plt.title('Model with noise')
        noise = np.random.multivariate_normal([0.,0.],random.choice(self.cov),len(JmKsingles))
        plt.scatter([JmKsingles[i]+noise[i][0] for i in range(len(JmKsingles))],[magJsingles[i]+noise[i][1] for i in range(len(JmKsingles))],s=1,edgecolors='none',c=[1.,0.,0.])
        plt.scatter([JmKdoubles[i]+noise[i][0] for i in range(len(magJdoubles))],[magJdoubles[i]+noise[i][1] for i in range(len(magJdoubles))],s=1,edgecolors='none',c=[0.,0.,1.])
        plt.scatter([JmKtriples[i]+noise[i][0] for i in range(len(JmKtriples))],[magJtriples[i]+noise[i][1] for i in range(len(JmKtriples))],s=1,edgecolors='none',c=[0.,0.6,.0])
        plt.xlim(self.JmKlims)
        plt.ylim(self.magJlims)
        plt.gca().invert_yaxis()
        plt.xlabel('$J-K$')
        #plt.ylabel('$M_J$')
        plt.tight_layout()
        plt.show()
    
    # scatter plot with marginalized single/double/triple posteriors, etc.
    def plotellipses(self,hyperparams):
        def ellipse(g,sigma=1.):
            cov = sigma**2.*g[1]
            degs = np.linspace(0.,2.*pi,1000)
            eigvals,eigvectors = np.linalg.eig(cov)
            x = g[0][0]+math.sqrt(eigvals[0])*eigvectors[0,0]*np.sin(degs)+math.sqrt(eigvals[1])*eigvectors[0,1]*np.cos(degs)
            y = g[0][1]+math.sqrt(eigvals[0])*eigvectors[1,0]*np.sin(degs)+math.sqrt(eigvals[1])*eigvectors[1,1]*np.cos(degs)
            return [x,y]
        alpha,beta_1,beta_2,gamma,binary_fraction,triples_fraction = hyperparams
        sg_cMs,sg_nums,sg_covs = self.single_gaussians(hyperparams,JmKlims=[0.,1.2])
        dg_cMs,dg_nums,dg_covs = self.double_gaussians(hyperparams)
        tg_cMs,tg_nums,tg_covs = self.triple_gaussians(hyperparams)
        print len(sg_cMs),len(dg_cMs),len(tg_cMs)
        for i in range(len(sg_cMs)):
            x,y = ellipse([sg_cMs[i],sg_covs[i]])
            plt.plot(x,y,color=[1.,0.,0.])
        for i in range(len(dg_cMs)):
            x,y = ellipse([dg_cMs[i],dg_covs[i]])
            plt.plot(x,y,color=[0.,0.,1.])
        for i in range(len(tg_cMs)):
            x,y = ellipse([tg_cMs[i],tg_covs[i]])
            plt.plot(x,y,color=[0.,0.6,0.])
        #plt.xlim(self.JmKlims)
        #plt.ylim(self.magJlims)
        plt.gca().invert_yaxis()
        plt.xlabel('$J-K$')
        #plt.ylabel('$M_J$')
        plt.tight_layout()
        plt.show()