import matplotlib.pyplot as plt
import numpy as np
import math
import random
pi=math.pi


class model():
    
    # Initiation function
    # thinningfactor is for thinnning out the sample, in case you want to run a quicker preliminary posterior or scatter plot
    # if fine=True, not as many gaussians in the binary and trinary population are thrown away
    def __init__(self,thinningfactor=1,fine=False):
        self.fine = fine
        npzfile = np.load("./Data/TGASx2MASS_cut_d<200pc.npz")
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
        #print np.max([math.sqrt(x[1][1]) for x in self.cov])
        if thinningfactor!=1:
            self.length = len(self.magJlist)
        self.median_sigmac = math.sqrt(np.median([c[0][0] for c in self.cov]))
        self.median_sigmaM = math.sqrt(np.median([c[1][1] for c in self.cov]))
        #plt.hist(parerrlist/parlist,100)
        #plt.xlabel('Parallax error / observed parallax')
        #plt.ylabel('Object count')
        #plt.show()
        #exit()
    
    # returns the data of a star in the sample
    def output_star(self,i=None):
        if i==None:
            return self.JmKlist,self.magJlist,self.cov
        else:
            return self.JmKlist[i],self.magJlist[i],self.cov[i]
    
    # compute a bivariate gaussian value
    def bivariate(self,x,cov):
        invnorm = cov[0,0]*cov[1,1]-cov[0,1]**2.
        cov_inv = [[cov[1,1]/invnorm,-cov[0,1]/invnorm],[-cov[0,1]/invnorm,cov[0,0]/invnorm]]
        cov_det = cov[0,0]*cov[1,1]-cov[0,1]**2.
        return math.exp(-1./2.*np.dot(np.dot(cov_inv,x),x))/(2.*pi*math.sqrt(cov_det))
    
    # adds upp a number of magnitudes
    def add_magnitudes(self,Mvec):
        return -2.5*np.log10( sum([10.**(-Mi/2.5) for Mi in Mvec]) )
    
    # input color and magnitude of two stars,
    # returns color and magnitude of the merged binary
    def merge(self,cM1,cM2):
        M_k1 = cM1[1]-cM1[0]
        M_k2 = cM2[1]-cM2[0]
        M_j,M_k = [self.add_magnitudes([cM1[1],cM2[1]]),self.add_magnitudes([M_k1,M_k2])]
        return np.array([M_j-M_k,M_j])
    
    # add two gaussian distributions for single stars together as a binary
    # this one operates in (c,M_J)-space
    # g1 and g2 should be on form [[mean(c),mean(M_J)], covariance matrix]
    def gaussian_sum(self,cM1,cov1,cM2,cov2):
        mean = self.merge(cM1,cM2)
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
        JmKs = np.linspace(JmKlims[0],JmKlims[1],int((JmKlims[1]-JmKlims[0])/0.0499)+1)
        def cov(JmK):
            JmKwidth = 0.05
            eig1 = np.array([1.,beta_2])
            eig2 = np.array([beta_2,-1.])
            eigs = np.transpose(np.array([eig1,eig2]))
            eigs_inv = np.linalg.inv(eigs)
            eigvals = np.matrix( [[(beta_2**2.+1.)*JmKwidth**2.,0.],[0.,gamma**2.]] )
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
            for j in range(i,len(sg_cMs),2-int(self.fine)):
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
            for j in range(i,len(sg_cMs),4-int(self.fine)):
                for k in range(j,len(sg_cMs),4-int(self.fine)):
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
    
    # computes the normalization fraction \bar{N}
    def inclusion_fraction(self,cM,cov):
        res = 1./4.*( math.erf((self.magJlims[1]-cM[1])/math.sqrt(2.*(cov[1][1]+self.median_sigmaM**2.))) \
                    -math.erf((self.magJlims[0]-cM[1])/math.sqrt(2.*(cov[1][1]+self.median_sigmaM**2.))))* \
                    ( math.erf((self.JmKlims[1]-cM[0])/math.sqrt(2.*(cov[0][0]+self.median_sigmac**2.))) \
                    -math.erf((self.JmKlims[0]-cM[0])/math.sqrt(2.*(cov[0][0]+self.median_sigmac**2.))))
        return res
    
    # generate all the gaussians of the gaussian mixture, given populationparameters
    def generate_model_gaussians(self,hyperparams):
        alpha,beta_1,beta_2,gamma,binary_fraction,triples_fraction = hyperparams
        sg_cMs,sg_nums,sg_covs = self.single_gaussians(hyperparams)
        norm_single = np.sum( [sg_nums[i]*self.inclusion_fraction(sg_cMs[i],sg_covs[i]) for i in range(len(sg_cMs))] )
        dg_cMs,dg_nums,dg_covs = self.double_gaussians(hyperparams)
        norm_double = np.sum( [dg_nums[i]*self.inclusion_fraction(dg_cMs[i],dg_covs[i]) for i in range(len(dg_cMs))] )
        tg_cMs,tg_nums,tg_covs = self.triple_gaussians(hyperparams)
        norm_triple = np.sum( [tg_nums[i]*self.inclusion_fraction(tg_cMs[i],tg_covs[i]) for i in range(len(tg_cMs))] )
        gaussians = [[sg_cMs,sg_nums,sg_covs],[dg_cMs,dg_nums,dg_covs],[tg_cMs,tg_nums,tg_covs]]
        #print len(sg_cMs),len(dg_cMs),len(tg_cMs)
        norms = np.array([norm_single,norm_double,norm_triple])
        fractions = np.array( [(1.-binary_fraction-triples_fraction),binary_fraction,triples_fraction] )
        return gaussians,norms,fractions
    
    # the log posterior on population parameters
    def hyperposterior(self,hyperparams):
        alpha,beta_1,beta_2,gamma,binary_fraction,triples_fraction = hyperparams
        if binary_fraction<0. or binary_fraction>1. or triples_fraction<0. or triples_fraction>1. or gamma<0.:
            return -np.inf
        def priors(hyperparams):
            alpha,beta_1,beta_2,gamma,binary_fraction,triples_fraction = hyperparams
            return 0.#-1./2.*(binary_fraction/0.3)**2. -1./2.*(triples_fraction/0.1)**2.# 
        gaussians,norms,fractions = self.generate_model_gaussians(hyperparams)
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
        if binary_fraction<0. or binary_fraction>1. or triples_fraction<0. or triples_fraction>1. or gamma<0.:
            return -np.inf
        gaussians,norms,fractions = self.generate_model_gaussians(hyperparams)
        res = []
        for i in range(self.length):
            JmK_i = self.JmKlist[i]
            magJ_i = self.magJlist[i]
            cov_i = self.cov[i]
            subres = np.array( [0.,0.,0.] )
            for k in range(3):
                for i_g in range(len(gaussians[k][0])):
                    mudiff = [JmK_i-gaussians[k][0][i_g][0],magJ_i-gaussians[k][0][i_g][1]]
                    if abs(mudiff[0])<0.2 and abs(mudiff[1])<.8:
                        cov_sum = cov_i+gaussians[k][2][i_g]
                        subres[k] += fractions[k]*gaussians[k][1][i_g]*self.bivariate(mudiff,cov_sum)/norms[k]
            res.append(subres)
        return np.array( res )
    
    #infer the object type
    def infer_obj_type(self,hyperparamsvec,path=None):
        single_posts = np.zeros([self.length,3])
        for hyp in hyperparamsvec:
            print hyp
            single_posts += np.array(self.posteriors(hyp))
        if path!=None:
            np.save(path,[[sp[0]/sum(sp),sp[1]/sum(sp),sp[2]/sum(sp)] for sp in single_posts])
        return [[sp[0]/sum(sp),sp[1]/sum(sp),sp[2]/sum(sp)] for sp in single_posts]
    
    # scatter plot with marginalized single/double/triple posteriors, etc.
    def scatter(self,hyperparams,hyperparamsvec=None,dotsize=5,save=False,obj_types_file=None):
        alpha,beta_1,beta_2,gamma,binary_fraction,triples_fraction = hyperparams
        if obj_types_file==None:
            if hyperparamsvec==None:
                single_posts = self.infer_obj_type(hyperparamsvec)
            #single_colors = [[sp[0]/sum(sp),0.6*sp[2]/sum(sp),sp[1]/sum(sp)] for sp in single_posts]
        else:
            single_posts = np.load(obj_types_file)
        single_colors = [[sp[1]/sum(sp),0.,sp[2]/sum(sp)] for sp in single_posts]    
        single_dists = []
        double_dists = []
        triple_dists = []
        """for i in range(self.length):
            if single_colors[i][0]>single_colors[i][1] and single_colors[i][0]>single_colors[i][2]:
                single_dists.append(1./self.parlist[i])
            elif single_colors[i][1]>single_colors[i][2]:
                double_dists.append(1./self.parlist[i])
            else:
                triple_dists.append(1./self.parlist[i])"""
        import matplotlib.gridspec as gridspec
        import matplotlib as mpl
        from matplotlib.colors import LinearSegmentedColormap
        f = plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.,1.,1.,0.1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])
        cmap = LinearSegmentedColormap.from_list('mycmap', [(0.0, [0.,0.,0.]),(0.5, [1.,0.,0.]),(1.0, [0.,0.,1.])])
        ax1.set_xticks([0.5,0.6,0.7])
        ax2.set_xticks([0.5,0.6,0.7])
        ax3.set_xticks([0.5,0.6,0.7,0.8])
        ax1.set_yticks([3.,3.5,4.,4.5,5.,5.5,6.])
        ax2.set_yticks([3.,3.5,4.,4.5,5.,5.5,6.])
        ax2.set_yticklabels(['','','','','','',''])
        ax3.set_yticks([3.,3.5,4.,4.5,5.,5.5,6.])
        ax3.set_yticklabels(['','','','','','',''])
        ax1.set_title('Data')
        ax2.set_title('Model')
        ax3.set_title('Model with noise')
        ax1.set_xlim(self.JmKlims)
        ax1.set_ylim([self.magJlims[1],self.magJlims[0]])
        ax2.set_xlim(self.JmKlims)
        ax2.set_ylim([self.magJlims[1],self.magJlims[0]])
        ax3.set_xlim(self.JmKlims)
        ax3.set_ylim([self.magJlims[1],self.magJlims[0]])
        ax1.set_xlabel('$J-K$')
        ax1.set_ylabel('$M_J$')
        ax2.set_xlabel('$J-K$')
        ax3.set_xlabel('$J-K$')
        from matplotlib import cm
        ax1.scatter(self.JmKlist,self.magJlist,s=dotsize,c=single_colors,edgecolors='none',vmin=0., vmax=1.)
        JmKsingles,magJsingles = self.generate_objects(4000*(1.-binary_fraction-triples_fraction),hyperparams)
        JmKdoubles,magJdoubles = self.generate_objects(4000*binary_fraction,hyperparams,obj_type='double')
        JmKtriples,magJtriples = self.generate_objects(4000*triples_fraction,hyperparams,obj_type='triple')
        ax2.scatter(JmKsingles,magJsingles,s=dotsize,edgecolors='none',c=[0.,0.,0.])
        ax2.scatter(JmKdoubles,magJdoubles,s=dotsize,edgecolors='none',c=[1.,0.,0.])
        ax2.scatter(JmKtriples,magJtriples,s=dotsize,edgecolors='none',c=[0.,0.,1.])
        covs_noise = []
        for i in range(6):
            covs_sub = []
            for j in range(len(self.magJlist)):
                if self.magJlist[j]>3.+float(i)/2. and self.magJlist[j]<3.+float(i+1)/2.:
                    covs_sub.append(self.cov[j])
            covs_noise.append(covs_sub)
        def noise(magJ):
            m = int(2.*(magJ-3.))
            if m>5:
                m=5
            if m<0:
                m=0
            cov = random.choice(covs_noise[m])
            return np.random.multivariate_normal([0.,0.],cov)
        noise_singles = [noise(magJ) for magJ in magJsingles]
        noise_doubles = [noise(magJ) for magJ in magJdoubles]
        noise_triples = [noise(magJ) for magJ in magJtriples]
        vs = ax3.scatter([JmKsingles[i]+noise_singles[i][0] for i in range(len(JmKsingles))],[magJsingles[i]++noise_singles[i][1] for i in range(len(JmKsingles))],s=dotsize,edgecolors='none',c=np.zeros(len(JmKsingles)),cmap=cmap,vmin=0.,vmax=1.)
        ax3.scatter([JmKdoubles[i]+noise_doubles[i][0] for i in range(len(magJdoubles))],[magJdoubles[i]+noise_doubles[i][1] for i in range(len(magJdoubles))],s=dotsize,edgecolors='none',c=[1.,0.,0.])
        ax3.scatter([JmKtriples[i]+noise_triples[i][0] for i in range(len(JmKtriples))],[magJtriples[i]+noise_triples[i][1] for i in range(len(JmKtriples))],s=dotsize,edgecolors='none',c=[0.,0.,1.])
        plt.gca().invert_yaxis()
        plt.subplots_adjust(wspace=0.1, hspace=0.0)
        norm = mpl.colors.Normalize(vmin=0., vmax=1.)
        clb = mpl.colorbar.ColorbarBase(ax4, cmap=cmap, norm=norm, orientation='vertical', ticks=[0.0, 0.5, 1.0])
        clb.set_ticklabels(['Single', 'Binary', 'Trinary'])
        if save:
            plt.savefig('./Figures/scatter.pdf')
        plt.show()
    
    # given stellar object i, return the (not normalized) posterior probability of the component stars
    # given that the object is binary or trinary (4 or 6 input params)
    # params are color1,mag1,color2,mag2(,color3,mag3)
    def disentangle(self,i,params,hyperparams):
        if len(params)==4:
            if params[0]>params[2] or params[2]-params[0]>1.:
                return 0.
            merged_obj = self.merge([params[0],params[1]],[params[2],params[3]])
            mudiff = [self.JmKlist[i]-merged_obj[0],self.magJlist[i]-merged_obj[1]]
            res3 = 1.
        elif len(params)==6:
            if params[0]>params[2] or params[2]>params[4] or params[4]-params[0]>1.:
                return 0.
            merged_obj = self.merge([params[0],params[1]],[params[2],params[3]])
            merged_obj = self.merge([merged_obj[0],merged_obj[1]],[params[4],params[5]])
            mudiff = [self.JmKlist[i]-merged_obj[0],self.magJlist[i]-merged_obj[1]]
            res3 = 0.
        res1 = 0.
        res2 = 0.
        sg_cMs,sg_nums,sg_covs = self.single_gaussians(hyperparams,JmKlims=[0.2,1.8])
        for k in range(len(sg_cMs)):
            if abs(params[0]-sg_cMs[k][0])<0.2:
                res1 += self.bivariate([params[0]-sg_cMs[k][0],params[1]-sg_cMs[k][1]],sg_covs[k])
            if abs(params[2]-sg_cMs[k][0])<0.2:
                res2 += self.bivariate([params[2]-sg_cMs[k][0],params[3]-sg_cMs[k][1]],sg_covs[k])
            if len(params)==6 and abs(params[4]-sg_cMs[k][0])<0.2:
                res3 += self.bivariate([params[4]-sg_cMs[k][0],params[5]-sg_cMs[k][1]],sg_covs[k])
        return res1*res2*res3*math.exp(-hyperparams[0]*merged_obj[0])*self.bivariate(mudiff,self.cov[i])
