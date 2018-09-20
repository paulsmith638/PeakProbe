import sys,os
import numpy as np

"""
generic stats and utility functions
independent of scipy, import carefully
"""
class StatFunc:
    def assign_bin(self,resolution):
        #assigns a bin number based on resolution (log based bins)
        bins = {'bin1':[0.00,1.31],'bin2':[1.31,1.50],'bin3':[1.50,1.72],'bin4':[1.72,1.97],'bin5':[1.97,2.25],
                'bin6':[2.25,2.58],'bin7':[2.58,2.95],'bin8':[2.95,3.37],'bin9':[3.37,np.inf]}

        binno = np.zeros(resolution.shape[0],dtype=np.int16)
        for binname,resrange in bins.iteritems():
            binint = int(binname[3])
            lowc,highc = resrange[0],resrange[1]
            binl_sel = resolution >= lowc
            binh_sel = resolution < highc
            bin_sel = np.logical_and(binl_sel,binh_sel)
            binno[bin_sel] = binint
        return binno

    def spline_basis(self,knotn,knotk,xval):
        #taken from http://www.stat.columbia.edu/~madigan/DM08/regularization.ppt.pdf
        #returns dk(X)
        #knotn is segment knot, knotk is outer (right) knot
        #putmask faster than np.clip()
        xshiftn = xval-knotn
        xshiftk = xval-knotk
        np.putmask(xshiftn,xshiftn < 0.0,0)
        np.putmask(xshiftk,xshiftk < 0.0,0)
        #dkX = (np.clip(xval-knotn,0,np.inf)**3 - np.clip(xval-knotk,0,np.inf)**3)/(knotk-knotn)
        #inline multiplication about 2x faster than np.pow or **
        dkX = (xshiftn*xshiftn*xshiftn - xshiftk*xshiftk*xshiftk)/(knotk-knotn)
        return dkX

    def spline4k(self,param,xval):
        #knot x values
        #6 evenly spaced knots between 1 and 4A.
        knots = np.linspace(1.0,4.0,6)
        xval = np.array(xval).reshape(-1).astype(np.float64)
        #cubic spline regression on 1.0,x,and natural cubic spline basis functions
        linear_terms = param[0]*np.ones(xval.shape[0])+param[1]*xval
        num_dterms = knots.shape[0] - 2
        dterms = np.zeros(xval.shape[0],dtype=np.float64)
        for i in range(num_dterms):
            basis=self.spline_basis(knots[i],knots[-1],xval) - self.spline_basis(knots[-2],knots[-1],xval)
            dterms=dterms+param[i+2]*basis
        net_spline = linear_terms + dterms
        return net_spline


    def johnsonsu_stats(self,param):
        #in wikipedia notation, xi is loc, lambda is scale, delta is b, gamma a
        a,b,loc,scale = param
        xi = loc
        lam = np.clip(scale,0.001,np.inf) #must be greater than zero, fitting may go wrong
        delta = np.clip(b,0.05,np.inf) # avoid exp overruns,must be greater than zero
        gamma = a
        pdf_mean = self.jsu_mean((gamma,delta,xi,lam))
        pdf_var = self.jsu_var((gamma,delta,xi,lam))
        return pdf_mean,pdf_var

    def jsu_mean(self,param):
        gamma,delta,xi,lam = param
        mean = (xi - lam*np.exp((delta**-2)/2)*np.arcsinh(gamma/delta))
        return mean
    def jsu_var(self,param):
        gamma,delta,xi,lam = param #xi is not actually needed (shift parameter)
        var = ((lam**2)/2.0)*(np.exp(delta**-2)-1.0)*(np.exp(delta**-2)*np.cosh((2.0*gamma)/delta) + 1.0)
        return var.astype(np.float32)

    def johnsonsu_pdf(self,xval,a,b,loc,scale):
        #borrowed from scipy 
        scale = np.clip(scale,0.001,np.inf) # must be greater than xero
        b = np.clip(b,0.05,np.inf) # must be greater than zero
        xshift = (xval-loc)/scale
        prefactor1 = b/((scale*np.sqrt(2*np.pi))*(np.sqrt(1+xshift**2)))
        prob = prefactor1*np.exp(-0.5*(a+b*np.arcsinh(xshift))**2)
        return prob.astype(np.float32)


    #pdf for mean zero, var = 1 normal data
    def norm_pdf(self,xdata):
        prefact = 1.0/np.sqrt(2*np.pi)
        ydat = np.exp(-np.multiply(xdata,xdata)/2.0) 
        pdfy = (prefact*ydat).astype(np.float32)
        return pdfy


    def llr_to_prob(self,llr_x):#logit
        return  (1.0/(np.exp(-llr_x) + 1.0)).astype(np.float32)


    def fishers_disc(self,resolution):
        #returns Fishers_disc for PCA data
        pca_view_col = ['RX0','RX1','RX2','RX3','RX4','RX5',
                        'RX6','RX7','RX8','RX9','RX10','RX11',
                        'RX12','RX13','RX14','RX15','RX16','RX17','RX18']

        sn_data = []
        for index,column in enumerate(pca_view_col):
            sfit,wfit = self.get_jsu_coeffs(index,resolution)
            s_means,s_var = self.johnsonsu_stats(sfit)
            w_means,w_var = self.johnsonsu_stats(wfit)
            sn=((s_means-w_means)**2)/(s_var + w_var)
            sn_data.append(sn)
        return np.array(sn_data)


    #CHI SQ 
    def chisq_pdf(self,k,xval): #conventional chisq
    #allow non-discrete dof
        #avoiding scipy 
        gamma_k2 = np.math.gamma(k/2.0)
        chisq = 1/(2**(k/2.0)*(gamma_k2))*xval**((k/2.0)-1)*np.exp(-xval/2.0)
        return chisq     

    def chisq_cdf_k2(self,xval):
        #simple for 2-dof
        return 1.0 - np.exp(-xval/2.0)


    def chisq_cdf(self,xval,k):
        #hacked quick integration, quad.spi doesn't do well here, so wing it
        numbins = 10000
        chibins = np.linspace(0,200,numbins) #ad hoc
        binwidth = float(np.amax(chibins))/numbins
        chidensity = self.chisq_pdf(k,chibins)*binwidth
        values = []
        for x in xval:
            tosumbin = chibins <= x #boolean mask
            cdfsum = np.nansum(chidensity[tosumbin])
            values.append(cdfsum)
        return np.clip(np.array(values),0.0,1.0)

    def cmatrix(self,label,pred,npop=4):
        #confusion matrix for scoring
        #expects integer >= 1 for label/pred
        max_all = npop+1
        cmatrix = np.zeros((max_all,max_all),dtype=np.int32)
        for i in np.arange(1,max_all,1):
            for j in np.arange(1,max_all,1):
                cmatrix[i,j] = np.count_nonzero(np.logical_and(label==i,pred==j))
        cmatrix[:,0] = np.nansum(cmatrix,axis=1)
        cmatrix[0,:] = np.nansum(cmatrix,axis=0)
        return cmatrix
