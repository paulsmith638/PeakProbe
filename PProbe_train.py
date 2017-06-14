#cannot be run through phenix python, as scipy conflicts
#must use database output and run in regular python2.7
import sys,os,copy,math,ast
import numpy as np
#from numpy import linalg as LA
#sys.path.append('/usr/lib64/python2.7/site-packages')
import scipy as sp
import scipy.optimize as op
import scipy.integrate as spi
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
import sqlite3 as lite
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from PProbe_classify import ClassifierFunctions as ppclas

#constant for training
RESMIN = 0.7
RESMAX = 4.2

#functions used in resolution dependent scaling schemes
class TrainingFunctions:
     def __init__(self,verbose=False):
          cfunc = ppclas(verbose=True)
          #function from classifier needed here
          #pass references to generate unbound methods
          self.johnsonsu_stats=cfunc.johnsonsu_stats
          self.johnsonsu_pdf=cfunc.johnsonsu_pdf
          self.get_stats=cfunc.get_stats
          self.get_res_scales=cfunc.get_res_scales
          self.get_post_pca_scales = cfunc.get_post_pca_scales
          self.get_jsu_coeffs = cfunc.get_jsu_coeffs
          self.gen_xform_mat = cfunc.gen_xform_mat
          self.xform_data = cfunc.xform_data
          self.standardize_data = cfunc.standardize_data
          self.pca_xform_data = cfunc.pca_xform_data
          self.discriminant_analysis = cfunc.discriminant_analysis
          self.ppsel = cfunc.ppsel #inherit imported class?

     def johnsonsu_target(self,param,xval,yval):
          a,b,loc,scale = param
          #include L2 norm, weight on errors empirical
          return np.dot(param,param)+100.0*np.sum((yval-self.johnsonsu_pdf(xval,a,b,loc,scale))**2)

     def fit_jsu(self,xval,yval):
          initial = np.array((1.0,1.0,np.nanmean(xval),1.0))
          #constraints = np.array([(0.01,10.0),(-50.0,50.0),(-50.0,50.0),(ymin-3.0*yrange,ymax+3.0*yrange),
          #                   (ymin-3.0*yrange,ymax+3.0*yrange),(-yrange/1.0,yrange/1.0),(-yrange/1.0,yrange/1.0)])
          L = op.fmin_bfgs(self.johnsonsu_target,initial,args=(xval,yval))
          return L
     

     # Chisq functions not currently used in workflow, but there to check
     # distributions of feature vector Euclidian distances during rejection 
     # cycles -- as misidentified peaks are rejected, the distribution of 
     # these distances in the kernel matrix converged nicely to a chisq
     # distribution
     def chisq_pdf(self,param,xval):
          k,loc,scale = param
          #allow non-discrete dof
          gamma_int = lambda x,z: (x**(z-1))*np.exp(-x)
          gamma_k2 = spi.quad(gamma_int,0,np.inf,args=(k/2.0,))[0]
          xval = (xval-loc)/scale
          chisq = 1/(2**(k/2.0)*(gamma_k2))*xval**((k/2.0)-1)*np.exp(-xval/2.0)
          return chisq/scale

     def chisq_cdf(self,param,xval):
          #terrible hack at numerical integration and iteration
          numbins = 1000
          values = []
          for x in xval:
               bins = np.linspace(0,x,numbins)
               binwidth = float(x)/numbins
               cdfsum = np.nansum(self.chisq_pdf(param,bins)*binwidth)
               values.append(cdfsum)
          return np.array(values)
          
     def chisq_target(self,param,xval,yval):
          target=np.nansum((yval - self.chisq_pdf(param,xval))**2)
          return target

     def chisq_fit(self,xval,yval):
          initial = np.array((1.0,0.0,1.0))#initial dof,loc,scale guesses
          constraints = np.array([(3.0,99.0),(0.0,100),(0.1,99.0)])
          L = op.fmin_l_bfgs_b(self.chisq_target,initial,args=(xval,yval),bounds=constraints,approx_grad=True)
          xtest = np.linspace(0,1000,1000)
          print "DIST STAT",np.nansum(self.chisq_pdf(L[0],xtest)),np.nansum(np.multiply(self.chisq_pdf(L[0],xtest),xtest))
          return (L[0],np.nansum(np.multiply(self.chisq_pdf(L[0],xtest),xtest)))


     """
     SPLINE fitting, currently fixed at 4 internal knots
     """

     #@profile
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
          knots = np.linspace(1.0,RESMAX,6)
          #cubic spline regression on 1.0,x,and natural cubic spline basis functions
          linear_terms = param[0]*np.ones(xval.shape[0])+param[1]*xval
          num_dterms = knots.shape[0] - 2
          dterms = np.zeros(xval.shape[0])
          for i in range(num_dterms):
               basis=self.spline_basis(knots[i],knots[-1],xval) - self.spline_basis(knots[-2],knots[-1],xval)
               dterms=dterms+param[i+2]*basis

          return linear_terms + dterms

          
     def spline4k_target(self,param,xval,yval):
          #return np.dot(param,param)**2 + 1000.0*np.nansum((yval - self.spline4k(param,xval))**2)
          return np.nansum(np.abs(param)) + 1000.0*np.nansum((yval - self.spline4k(param,xval))**2) 


     def spline4k_fit(self,xval,yval,augment=False):
          initial = np.ones(6)
          if augment:
               axval = np.zeros(xval.shape[0] + 1)
               ayval = np.zeros(xval.shape[0] + 1)
               for index in range(xval.shape[0]):
                    axval[index] = xval[index]
                    ayval[index] = yval[index]
               axval[-1] = 5.0 #dummy point at 5.0A to keep splines flat
               ayval[-1] = yval[-1]
               xval = axval
               yval = ayval
          L = op.fmin_bfgs(self.spline4k_target,initial,args=(xval,yval))
          return L

     #appends point and start and end of array equal to lowest/highest value
     #at 0.5 and 5.0 resolution to keep spline fitting from going off the rails
     #would be better to use a spline with defined slopes past the knots

     def augment_array(self,array,is_res=False):
          return array
          if len(array.shape) == 2:
               new_array = np.zeros((array.shape[0],array.shape[1]+2),dtype=np.float64)
               if is_res:
                    new_array[:,0] = np.ones(array.shape[1])*0.5
                    new_array[:,-1] = np.ones(array.shape[1])*5.0
               else:
                    new_array[:,0] = array[:,0]
                    new_array[:,-1] = array[:,-1]
               for column in range(array.shape[0]):
                    new_array[:,column+1] = array[:,column]

          if len(array.shape) == 1:
               array_size = array.shape[0] + 2
               new_array = np.zeros(array_size,dtype=np.float64)
               if is_res:
                    new_array[0] = 0.5
                    new_array[-1] = 5.0
               else:
                    new_array[0] = array[0]
                    new_array[-1] = array[-1]
               for column in range(array.shape[0]):
                    new_array[column+1] = array[column]
          #return new_array

               
          


     """
     functions for training
     calculate and write out model coefficients for
     1) resolution dep raw scaling
     2) resolution dep pca transformation
     3) resolution dep scaling of pca xformed data
     4) resolution dep jsu pdf coefficients for discriminant analysis
     """

     def calculate_res_scales(self,data_array,post_pca=False,plot=False):
          """
          calculates mean/sigma in a resolution variant scheme for a given training set
          data array must be numpy structured array with correct feature names
          """
          scales_dict = {}
          selectors = self.ppsel(data_array)
          if post_pca:
               col_names=selectors.pca_view_col
          else:
               col_names = selectors.std_view_col
          print "RESSCALE",col_names
          res = data_array['res']
          if plot:
               gridplot = plt.figure(figsize=(24,16))
          for index,column in enumerate(col_names):
               data_column = data_array[column]
               col_res,col_smean,col_wmean,col_sstd,col_wstd,col_std = [],[],[],[],[],[]
               for population,selector in zip(('obss','obsw','all'),(selectors.inc_obss_bool,
                                                                     selectors.inc_obsw_bool,
                                                                     selectors.included_data_bool)):
                    input_data = data_column[selector]
                    input_res = res[selector]
                    binstep = 0.25
                    #40 resolution windows frm resmin to resmax
                    for i in np.linspace(RESMIN,RESMAX,40):
                         binlr_sel = np.apply_along_axis(lambda x: x > i,0,input_res)
                         binhr_sel = np.apply_along_axis(lambda x: x <= binstep+i,0,input_res)
                         binres_sel = np.logical_and(binlr_sel,binhr_sel)
                         input_data_bin=input_data[binres_sel]
                         res_bin=input_res[binres_sel]
                         #if at least 10 peaks in the bin
                         if input_data_bin.shape[0] > 10:
                              bin_resmean = np.nanmean(res_bin)
                              #get std from entire population
                              if population == "all":
                                   col_res.append(bin_resmean)
                                   col_std.append(np.nanstd(input_data_bin))
                              #mean of obsss
                              if population == "obss":
                                   col_smean.append(np.nanmean(input_data_bin))
                                   col_sstd.append(np.nanstd(input_data_bin))
                              #mean of obsw
                              if population == "obsw":
                                   col_wmean.append(np.nanmean(input_data_bin))
                                   col_wstd.append(np.nanstd(input_data_bin))
                         #very few peaks at low resolution, extend data as needed from previous bins
                         else:
                              if population == "all":
                                   try:
                                        col_res.append(col_res[-1])
                                        col_std.append(col_std[-1])
                                   except:
                                        col_std.append(1.0)
                                        try:
                                             col_res.append(col_res[-1])
                                        except:
                                             col_res.append(1.0)
                              if population == "obss":
                                   try:
                                        col_smean.append(col_smean[-1])
                                        col_sstd.append(col_sstd[-1])
                                   except:
                                        col_smean.append(1.0)
                                        col_sstd.append(1.0)
                              if population == "obsw":
                                   try:
                                        col_wmean.append(col_wmean[-1])
                                        col_wstd.append(col_wstd[-1])
                                   except:
                                        col_wmean.append(1.0)
                                        col_wstd.append(1.0)
               #data is calculated, now analyze
               #convert to np array
               binstd = np.array(col_std)
               binres = np.array(col_res)
               mean_sval = np.array(col_smean)
               std_sval = np.array(col_sstd)
               mean_wval = np.array(col_wmean)
               std_wval = np.array(col_wstd)
               #take bin mean as average of s and w populations (centers data)
               binmean = np.divide(mean_wval + mean_sval,2.0)
               mean_spline_coeff = self.spline4k_fit(binres,binmean,augment=True)
               #sig must be positive, fit log sigma
               log_sig_spline_coeff = self.spline4k_fit(binres,np.log(binstd),augment=True)
               scales_dict[column] = (tuple(mean_spline_coeff),tuple(log_sig_spline_coeff))
               print "MEAN SPLINE",column,mean_spline_coeff
               print "logSIG SPLINE",column,log_sig_spline_coeff

               #plotting functions to test fit
               if plot:
                    sub = gridplot.add_subplot(8,5,index+1)
                    plot_xval = np.linspace(0.5,5.0,100)
                    mean_plot_spline = self.spline4k(mean_spline_coeff,plot_xval)
                    sig_plot_spline = np.exp(self.spline4k(log_sig_spline_coeff,plot_xval))
                    plt.title(column)
                    sub.plot(plot_xval,mean_plot_spline,'k-')
                    sub.plot(plot_xval,sig_plot_spline,'c-')  
                    sub.scatter(binres,mean_wval,color='blue')
                    sub.scatter(binres,mean_sval,color='red')
                    sub.scatter(binres,binmean,color='black')
                    sub.scatter(binres,binstd,color='cyan')
                    sub = gridplot.add_subplot(8,5,index+21)
                    sn=np.divide((np.power(np.subtract(mean_sval,mean_wval),2)),binstd**2)
                    plt.title(column)
                    sub.scatter(binres,sn,color='green')
          if plot:
               if post_pca:
                    figname = "PCA_resfit.png"
               else:
                    figname = "DATA_resfit.png"
               plt.savefig(figname)
               plt.clf()
               plt.close()

          



          if post_pca:
               filename = "pprobe_post_pca_resscales.dict"
          else:
               filename = "pprobe_pre_pca_resscales.dict"
          f = open(filename,'w')
          f.write(str(scales_dict))
          f.close()

     def calc_res_pca(self,norm_data,plot=False):
          """
          calculates resolution dependent PCA matrices binwise
          """
          selectors = self.ppsel(norm_data)
          res = norm_data['res']
          col_list = selectors.std_view_col
          view_dtype = selectors.std_view_dtype
          #select numerical data
          selected_data = norm_data[col_list].view(selectors.raw_dtype)
          #calculate covariance
          cov_data = np.cov(selected_data[:],rowvar=False)
          #calculate PCA eigvect,eigval for entire datablock for reference
          pca_total_val,pca_total_vect = np.linalg.eigh(cov_data)
          #sorted indices of eigenvalues, high to low
          pca_total_argsort = np.argsort(pca_total_val)[::-1]
          #reference array of sorted eigenvectors (all data)
          ref_pca = pca_total_vect.T[pca_total_argsort]

          #calculte PCA components in chunks by resolution, or by size/bin
          data_size = selected_data.shape[0]
          resbins = 20
          resmin = RESMIN
          resmax=RESMAX
          #divide resolution into bins
          resblocks = np.linspace(resmin,resmax,resbins)
          eig_data = []
          binres = []
          for sbin_size in (500,):
               bin_mask,num_bins = self.calc_res_bins(norm_data,sbin_size)
               for overlap_bin in np.arange(0,num_bins+1,2):
                    ressel = np.logical_or(bin_mask == overlap_bin,bin_mask == (overlap_bin + 1))
                    databin = selected_data[ressel]
                    res_block=res[ressel]
                    print "CALC EIGv for DATABIN",overlap_bin,np.amin(res_block),np.amax(res_block),np.nanmean(res_block)
                    bincov = np.cov(databin.T)
                    eigval,eigvect = np.linalg.eigh(bincov)
                    binpca_argsort = np.argsort(eigval)[::-1]
                    eig_data.append(eigvect.T[binpca_argsort])
                    #get resolutions for each databin
                    resmean = np.nanmean(res_block)
                    binres.append(resmean)

          #creates a 3d array, x is resolution slice, y and z are binwise eigenvectors matrix
          input_pca = np.array(eig_data)
          #next, orient each bin of PCA eigenvectors
          #first eigv is aligned to reference PCA set
          oriented_pca = np.zeros(input_pca.shape)
          #iterate by resolution
          for pcabin,pca_comp in enumerate(input_pca):
               #reference is initially the PCA xfrom of entire dataset
               if pcabin == 0:
                    ref_fit = ref_pca
               #initialze a matrix, same size as the cov matrix
               new_pca = np.zeros(pca_comp.shape)
               #pool of vectors from which to pick "best" eigenvector by matching to reference pca set
               vector_pool = pca_comp
               #boolean mask array 
               selbool = np.ones(pca_comp.shape[1],dtype=np.bool_)
               #iterate through eigenvectors
               for index in np.arange(pca_comp.shape[1]):
                    ref_vect = ref_fit[index]
                    dotvect = np.zeros(vector_pool.shape[0])
                    #get dot product for each each eigenvector and the reference vector (abs)
                    for eigind,eigv in enumerate(vector_pool):
                         #absolute value gives best co-linearity in either orientation
                         dotvect[eigind] = np.abs(np.dot(ref_vect,eigv))
                         #gives best fit only in as given orientations
                         #dotvect[eigind] = np.dot(orig_ref[index],eigv)
                    #find the largest dot product (sort plus/minus later)
                    print "DOTS",dotvect
                    best_ind = np.argmax(dotvect)
                    print "REFV",ref_vect
                    print "BEST",vector_pool[best_ind],best_ind
                    new_pca[index] = vector_pool[best_ind]
                    selbool[best_ind] = False
                    #eliminate the used vector from the pool
                    vector_pool = vector_pool[selbool]
                    selbool = selbool[selbool]
                    #"flip" eigenvectors that are reversed w.r.t the reference vector
                    dotprod = np.dot(new_pca[index],ref_vect)
                    if dotprod < 0:
                         print "FLIPr",dotprod,ref_vect
                         print "FLIPb",dotprod,new_pca[index]
                         new_pca[index] = np.multiply(new_pca[index],-1.0)
                    #print "FLIPn",np.dot(new_pca[index],ref_vect),new_pca[index]
                    ref_vect = new_pca[index]
               oriented_pca[pcabin] = new_pca
               #set the reference pca array to current to enforce smoothness
               ref_fit = new_pca.copy()

          #fit the resolution dependent PCA components to a usual spline function
          #does outlier exclusion
          pca_coeffs = np.zeros((oriented_pca.shape[1],oriented_pca.shape[2],6))
          for eigvect in np.arange(oriented_pca.shape[1]):
               if plot:
                    gridplot = plt.figure(figsize=(24,8))
               for component in np.arange(oriented_pca.shape[2]):
                    xval = np.array(binres)
                    yval = np.array(oriented_pca[:,eigvect,component])
                    print "FITTING EIGv Component",eigvect,component
                    pca_coeffs[eigvect,component,:] = self.spline4k_fit(xval,yval,augment=True)
                    #reject 3 worst outliers, testing training  
                    #residuals = (yval - self.spline4k(pca_coeffs[eigvect,component,:],xval))**2
                    #exclude_worst3 = np.argsort(residuals)[0:-3]
                    #reject bad points (testing/training)
                    #xval = xval[exclude_worst3]
                    #yval = yval[exclude_worst3]
                    pca_coeffs[eigvect,component,:] = self.spline4k_fit(xval,yval,augment=True)

                    #plotting functions
                    if plot:
                         fitxval = np.linspace(0.5,5.0,100)
                         fityval = self.spline4k(pca_coeffs[eigvect,component,:],fitxval)
                         sub = gridplot.add_subplot(4,5,component+1)
                         plt.title(component)
                         sub.set_xlim([0.5,5.0])
                         sub.scatter(xval,yval)
                         sub.plot(fitxval,fityval)
                         #add line to indicate total pca value
                         sub.plot((0.5,5.0),(ref_pca[eigvect,component],ref_pca[eigvect,component]))
               if plot:
                    plt.savefig("PCA_eigv_reject_fits_"+str(eigvect)+".png")
                    plt.clf()
                    plt.close()
          #store coeffs in a dictionary and write
          pca_coeffs_dict = {}
          for i in range(pca_coeffs.shape[0]):
               for j in range(pca_coeffs.shape[1]):
                    pca_coeffs_dict[str(i)+"_"+str(j)] = tuple(pca_coeffs[i,j])
          f = open("pprobe_pca_matrix_coeff.dict",'w')
          f.write(str(pca_coeffs_dict))
          f.close()

     def calc_res_bins(self,data_array,so4_per_bin):
          #bins data, returns integer array mask
          #input data should be sorted by resolution
          array_size = data_array.shape[0]
          bin_mask=np.zeros(array_size,dtype=np.int16)
          binno = 0
          count = 0
          #start from the high(low) res end
          for index,ori in enumerate(data_array['ori'][::-1]):
               bin_mask[index] = binno
               if ori == "SO4" or ori == "PO4":
                    count = count + 1
               if count == so4_per_bin:
                    count = 0
                    binno = binno + 1
          #flip array as we started at the end
          bin_mask = np.flipud(bin_mask)
          for i in range(binno + 1):
               binsize = np.count_nonzero(data_array['res'][bin_mask == i])
               print "BIN %s SIZE %s RES %s" % (i,binsize,np.nanmean(data_array['res'][bin_mask == i]))
          return bin_mask,binno

               
          


     def calc_jsu_coeff(self,data_array,plot=False):
          """
          takes data array with flagged populations (sulfate/water)
          generates a histogram of each population and a resolution bin, 
          then fits a johnsonsu distribution to each population
          then calculates a function to fit the coefficients of the distribution
          as a function of resolution
          """
          selectors = self.ppsel(data_array)
          if plot:
               gridplot_s = plt.figure(figsize=(24,8))
               gridplot_w = plt.figure(figsize=(24,8))
               xplot=np.linspace(0.5,5.0,100)
          #initialize empty lists to store jsu coefficient data
          res_data = []
          sfit_data = []
          wfit_data = []
          res = data_array['res']
          data_size = data_array.shape[0]
          resmin = RESMIN
          resmax=RESMAX
          #divide resolution into bins
          #resblocks = np.linspace(resmin,resmax,resbins)
          #ad hoc resolution bins to keep bins large
          resblocks = np.array((resmin,1.2,1.4,1.7,2.0,2.3,2.5,2.7,2.9,3.1,3.3,5.0))
          bin_mask,num_bins = self.calc_res_bins(data_array,500)
          for index,i in enumerate(range(num_bins + 1)):
               #ressel = selectors.reso_select_mask(data_array,i,resblocks[index+1])
               ressel = bin_mask == i
               binav = np.nansum(res[ressel])
               selected_data = data_array[ressel]
               print "FITTING JSU PEAKS",selected_data.shape[0],i,binav
               sfit_list,wfit_list,res_list = self.pop_fit_jsu(selected_data,plot=plot)
               res_data.append(res_list)
               sfit_data.append(sfit_list)
               wfit_data.append(wfit_list)
               if i == resblocks[-2]:
                    res_data.append(res_list)
                    sfit_data.append(sfit_list)
                    wfit_data.append(wfit_list)

          #each list is the jsu coefficients for each column of data
          #repeat the last points at 4.0A to avoid excessive extrapolation
          #there isn't much data in this range
          #convert lists to np arrays
          all_res_data=np.array(res_data)
          all_sfit_data=np.array(sfit_data)
          all_wfit_data=np.array(wfit_data)
          col_list = selectors.pca_view_col
          jsu_coeff_dict = {}
          for index,column in enumerate(col_list):
               #get all data in res,coeff pairs, 8 sets of coeffs total
               xdata=all_res_data[:,index]
               ysdata_c1=all_sfit_data[:,index,0]
               ysdata_c2=all_sfit_data[:,index,1]
               ysdata_c3=all_sfit_data[:,index,2]
               ysdata_c4=all_sfit_data[:,index,3]
               ywdata_c1=all_wfit_data[:,index,0]
               ywdata_c2=all_wfit_data[:,index,1]
               ywdata_c3=all_wfit_data[:,index,2]
               ywdata_c4=all_wfit_data[:,index,3]
               #fit each coefficient to res dependent function
               sc1_fit = self.spline4k_fit(xdata,ysdata_c1,augment=True)
               sc2_fit = self.spline4k_fit(xdata,ysdata_c2,augment=True)
               sc3_fit = self.spline4k_fit(xdata,ysdata_c3,augment=True)
               sc4_fit = self.spline4k_fit(xdata,ysdata_c4,augment=True)
               column_name = "SC"+str(index)
               #store in dictionary
               jsu_coeff_dict[column_name+str("_jsuc1")] = tuple(sc1_fit)
               jsu_coeff_dict[column_name+str("_jsuc2")] = tuple(sc2_fit)
               jsu_coeff_dict[column_name+str("_jsuc3")] = tuple(sc3_fit)
               jsu_coeff_dict[column_name+str("_jsuc4")] = tuple(sc4_fit)
               #same for water
               wc1_fit = self.spline4k_fit(xdata,ywdata_c1,augment=True)
               wc2_fit = self.spline4k_fit(xdata,ywdata_c2,augment=True)
               wc3_fit = self.spline4k_fit(xdata,ywdata_c3,augment=True)
               wc4_fit = self.spline4k_fit(xdata,ywdata_c4,augment=True)
               column_name = "WC"+str(index)
               jsu_coeff_dict[column_name+str("_jsuc1")] = tuple(wc1_fit)
               jsu_coeff_dict[column_name+str("_jsuc2")] = tuple(wc2_fit)
               jsu_coeff_dict[column_name+str("_jsuc3")] = tuple(wc3_fit)
               jsu_coeff_dict[column_name+str("_jsuc4")] = tuple(wc4_fit)
               if plot:
                    #plot coefficients vs resolution along with fits
                    #sulfate distribution first
                    sc1_func=self.spline4k(sc1_fit,xplot)
                    sc2_func=self.spline4k(sc2_fit,xplot)
                    sc3_func=self.spline4k(sc3_fit,xplot)
                    sc4_func=self.spline4k(sc4_fit,xplot)
                    sub1=gridplot_s.add_subplot(4,5,index+1)
                    sub1.plot(all_res_data[:,index],all_sfit_data[:,index,0],'bo')
                    sub1.plot(all_res_data[:,index],all_sfit_data[:,index,1],'ro')
                    sub1.plot(all_res_data[:,index],all_sfit_data[:,index,2],'go')
                    sub1.plot(all_res_data[:,index],all_sfit_data[:,index,3],'mo')
                    sub1.plot(xplot,sc1_func,'b-')
                    sub1.plot(xplot,sc2_func,'r-')
                    sub1.plot(xplot,sc3_func,'g-')
                    sub1.plot(xplot,sc4_func,'m-')
                    sub1.set_title("S_"+str(index))
                    #waters next
                    wc1_func=self.spline4k(wc1_fit,xplot)
                    wc2_func=self.spline4k(wc2_fit,xplot)
                    wc3_func=self.spline4k(wc3_fit,xplot)
                    wc4_func=self.spline4k(wc4_fit,xplot)
                    sub2=gridplot_w.add_subplot(4,5,index+1)
                    sub2.plot(all_res_data[:,index],all_wfit_data[:,index,0],'bo')
                    sub2.plot(all_res_data[:,index],all_wfit_data[:,index,1],'ro')
                    sub2.plot(all_res_data[:,index],all_wfit_data[:,index,2],'go')
                    sub2.plot(all_res_data[:,index],all_wfit_data[:,index,3],'mo')
                    sub2.plot(xplot,wc1_func,'b-')
                    sub2.plot(xplot,wc2_func,'r-')
                    sub2.plot(xplot,wc3_func,'g-')
                    sub2.plot(xplot,wc4_func,'m-')
                    sub2.set_title("W_"+str(index))
          if plot:
               gridplot_s.savefig("jsu_coeff_s.png")
               gridplot_w.savefig("jsu_coeff_w.png")
               plt.close()

          f = open("pprobe_jsu_coeffs.dict",'w')
          f.write(str(jsu_coeff_dict))
          f.close()
  
    
     def sn_plot(self):
          #calculates Fisher's linear discriminant vs. resolution
          #based on the Jsu pdf's from the data
          gridplot = plt.figure(figsize=(24,8))
          xplot = np.linspace(0.5,5.0,100)
          sn_sum = np.zeros(xplot.shape)
          for index in range(19):
               s_pdf_coeff,w_pdf_coeff=self.get_jsu_coeffs(index,xplot)
               s_means,s_var = self.johnsonsu_stats(s_pdf_coeff)
               w_means,w_var = self.johnsonsu_stats(w_pdf_coeff)
               sn=((s_means-w_means)**2)/(s_var + w_var)
               sn_sum=sn_sum+sn
               sub = gridplot.add_subplot(4,5,index+1)
               #sub.set_ylim([0,8])
               sub.plot(xplot,sn)
               sub.set_title("SN_"+str(index))
          sub=gridplot.add_subplot(4,5,20)
          #total SN as addition?
          sub.plot(xplot,sn_sum)
          #sub.set_ylim([0,8])
          gridplot.savefig("sn_plot.png")
          plt.close()

     def pop_fit_jsu(self,data_array,plot=False):
          selectors = self.ppsel(data_array)
          obss_sel=selectors.inc_obss_bool
          obsw_sel=selectors.inc_obsw_bool
          res_data=data_array['res']
          col_list = selectors.pca_view_col
          if plot:
               gridplot = plt.figure(figsize=(24,8))
          sfit_list = []
          wfit_list = []
          res_list = []
          #adjust histogram binning for size of dataset
          bins = np.clip(int(data_array.shape[0]/10),20,150)
          for index,column in enumerate(col_list):
               input_column=data_array[column]
               #set sensible histogram limits
               smean = np.nanmean(input_column[obss_sel])
               sstd =  np.nanstd(input_column[obss_sel])
               wmean = np.nanmean(input_column[obsw_sel])
               wstd =  np.nanstd(input_column[obsw_sel])
               slow = smean - 3.0*sstd
               wlow = wmean - 3.0*wstd
               shigh = smean + 3.0*sstd
               whigh = wmean + 3.0*wstd
               obss_hist,obss_bins = np.histogram(input_column[obss_sel],bins=bins,density=True,range=(slow,shigh))
               obsw_hist,obsw_bins = np.histogram(input_column[obsw_sel],bins=bins,density=True,range=(wlow,whigh))
               #calculates average of bin edges for fitting
               sfit = self.fit_jsu(((obss_bins[:-1] + obss_bins[1:]) / 2.0),obss_hist)
               wfit = self.fit_jsu(((obsw_bins[:-1] + obsw_bins[1:]) / 2.0),obsw_hist)
               sfit_list.append(sfit)
               wfit_list.append(wfit)
               res_list.append(np.nanmean(data_array['res']))

               if plot: 
                    xdata=np.linspace(np.amin((slow,wlow))-1.0,np.amax((shigh,whigh))+1.0,100)
                    sub = gridplot.add_subplot(4,5,index+1)
                    sub.plot(xdata,self.johnsonsu_pdf(xdata,sfit[0],sfit[1],sfit[2],sfit[3]),'r-')
                    sub.plot(xdata,self.johnsonsu_pdf(xdata,wfit[0],wfit[1],wfit[2],wfit[3]),'b-')
                    sub.hist(input_column[obss_sel], normed=True, bins=bins,range=(slow,shigh),color="red")
                    sub.hist(input_column[obsw_sel], normed=True, bins=bins,range=(wlow,whigh),color="blue")

          if plot:
               plt_str = str(res_list[-1])[0:4]
               plt.savefig("JSU_FITS_"+plt_str+".png")
               plt.clf()
               plt.close()
          return sfit_list,wfit_list,res_list     

