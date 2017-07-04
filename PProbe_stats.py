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
        xval = np.array(xval).reshape(-1)
        #cubic spline regression on 1.0,x,and natural cubic spline basis functions
        linear_terms = param[0]*np.ones(xval.shape[0])+param[1]*xval
        num_dterms = knots.shape[0] - 2
        dterms = np.zeros(xval.shape[0])
        for i in range(num_dterms):
            basis=self.spline_basis(knots[i],knots[-1],xval) - self.spline_basis(knots[-2],knots[-1],xval)
            dterms=dterms+param[i+2]*basis

        return linear_terms + dterms


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
        return (xi - lam*np.exp((delta**-2)/2)*np.arcsinh(gamma/delta))

    def jsu_var(self,param):
        gamma,delta,xi,lam = param #xi is not actually needed (shift parameter)
        return ((lam**2)/2.0)*(np.exp(delta**-2)-1.0)*(np.exp(delta**-2)*np.cosh((2.0*gamma)/delta) + 1.0)


    def johnsonsu_pdf(self,xval,a,b,loc,scale):
        #borrowed from scipy 
        scale = np.clip(scale,0.001,np.inf) # must be greater than xero
        b = np.clip(b,0.05,np.inf) # must be greater than zero
        xshift = (xval-loc)/scale
        prefactor1 = b/((scale*np.sqrt(2*np.pi))*(np.sqrt(1+xshift**2)))
        prob = prefactor1*np.exp(-0.5*(a+b*np.arcsinh(xshift))**2)
        return prob


    #pdf for mean zero, var = 1 normal data
    def norm_pdf(self,xdata):
        prefact = 1.0/np.sqrt(2*np.pi)
        ydat = np.exp(-np.multiply(xdata,xdata)/2.0) 
        return prefact*ydat


    def llr_to_prob(self,llr_x):#logit
        return  1.0/(np.exp(-llr_x) + 1.0)  


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


