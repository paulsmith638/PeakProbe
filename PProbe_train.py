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
from PProbe_classify import ClassifierFunctions
from PProbe_stats import StatFunc
from PProbe_matrix import PCA

#functions used in resolution dependent scaling schemes
class TrainingFunctions:
     def __init__(self,verbose=False,fastmode=False):
          ppcf = ClassifierFunctions(verbose=True)
          ppstat = StatFunc()
          pppca = PCA()
          #function from classifier needed here
          #pass references to generate unbound methods
          self.johnsonsu_stats=ppstat.johnsonsu_stats
          self.jsu_mean = ppstat.jsu_mean
          self.jsu_var = ppstat.jsu_var
          self.johnsonsu_pdf=ppstat.johnsonsu_pdf
          self.get_stats=ppcf.get_stats
          self.spline_basis = ppstat.spline_basis
          self.spline4k = ppstat.spline4k
          self.modal_matrix = pppca.modal_matrix
          self.pca_by_bin = pppca.pca_by_bin
          self.eig_sort = pppca.eig_sort

          self.get_res_scales=ppcf.get_res_scales
          self.get_post_pca_scales = ppcf.get_post_pca_scales
          self.get_jsu_coeffs = ppcf.get_jsu_coeffs
          self.gen_xform_mat = ppcf.gen_xform_mat
          self.xform_data = ppcf.xform_data
          self.standardize_data = ppcf.standardize_data
          self.pca_xform_data = ppcf.pca_xform_data
          self.discriminant_analysis = ppcf.discriminant_analysis
          self.batch_da = ppcf.batch_da
          self.initialize_results = ppcf.initialize_results
          self.score_stats = ppcf.score_stats
          self.ppsel = ppcf.ppsel #inherit imported class?

          #option for speedup for code testing
          self.fastmode = fastmode
          
     """
     Functions for curve fitting, require scipy which doesn't play nice with phenix.python
     Use caution with imports
     """

     #JSU FITTING
     def johnsonsu_target(self,param,xval,yval):
          a,b,loc,scale = param
          #b(delta) and scale(lambda) are highly correlated
          #weight on errors empirical
          #jsu_variance unstable when delta(b) is small and gamma(a) is more than 1.0
          #l2 norm on b and scale
          l2 = (b*b)+(scale*scale)
          var_penalty = self.jsu_var(param)
          return 0.05*var_penalty+l2+100.0*np.sum((yval-self.johnsonsu_pdf(xval,a,b,loc,scale))**2)

     def johnsonsu_restarget(self,param,xval,yval,ivals):
          a,b,loc,scale = param
          rest = np.nansum((param-ivals)**2) #harmonic restraint
          var_penalty = self.jsu_var(param)
          return 0.05*var_penalty+10.0*rest+100.0*np.sum((yval-self.johnsonsu_pdf(xval,a,b,loc,scale))**2)

     def fit_jsu(self,xval,yval,ivals=None,restrain=False):
          #to enforce smoothness among highly correlated parameters,
          #pass initial values and restraints
          #initial values is 4xfloat
          #if restraints is True, values are restrained to these initial values
          if ivals is not None:
               initial = np.array(ivals)
          else:
               initial = np.array((0.0,1.0,xval[np.argmax(yval)],1.0))
          #constraints are absolute
          constraints = np.array([(-999.0,999.0),(0.5,999.0),(-999.0,999.0),(0.05,999.0)])
          if restrain and ivals is not None:
               L = op.fmin_l_bfgs_b(self.johnsonsu_restarget,initial,args=(xval,yval,initial),
                                    bounds=constraints,approx_grad=True,disp=False)
          else:
               L = op.fmin_l_bfgs_b(self.johnsonsu_target,initial,args=(xval,yval),
                                    bounds=constraints,approx_grad=True,disp=False)
          return L[0] #just return fitted parameters
     

     """
     SPLINE fitting, currently fixed at 4 internal knots
     """

     def spline4k_target(self,param,xval,yval):
          #includes L1 norm
          return np.nansum(np.abs(param)) + 1000.0*np.nansum((yval - self.spline4k(param,xval))**2) 


     def spline4k_fit(self,xval,yval,augment=False):
          initial = np.ones(6)
          if augment:
               #appends point and start and end of array equal to lowest/highest value
               #at 0.3 and 5.0 resolution to keep spline fitting from going off the rails
               #would be better to use a spline with defined slopes past the knots

               axval = np.zeros(xval.shape[0] + 2)
               ayval = np.zeros(xval.shape[0] + 2)
               for index in np.arange(1,xval.shape[0]+1,1):
                    axval[index] = xval[index-1]
                    ayval[index] = yval[index-1]
               axval[0] = 0.6
               axval[-1] = 5.0 
               ayval[0] = yval[0]
               ayval[-1] = yval[-1]
               xval = axval
               yval = ayval
          L = op.fmin_bfgs(self.spline4k_target,initial,args=(xval,yval),disp=False)
          return L

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
          print "RESOLUTION DEPENDENT SCALING"
          res = data_array['res']
          if plot:
               gridplot = plt.figure(figsize=(24,16))
          bin_mask,num_bins = self.calc_res_bins(data_array,30,target_res_bin_width=0.075)
          for index,column in enumerate(col_names):
               data_column = data_array[column]
               col_res,col_smean,col_wmean,col_sstd,col_wstd,col_std,col_tmean = [],[],[],[],[],[],[]
               for population,selector in zip(('obss','obsw','all'),(selectors.inc_obss_bool,
                                                                     selectors.inc_obsw_bool,
                                                                     selectors.included_data_bool)):
                    input_data = data_column[selector]
                    input_binmask = bin_mask[selector]
                    input_res = res[selector]
                    for i in np.arange(num_bins + 1):
                         input_data_bin=input_data[input_binmask == i]
                         res_bin=input_res[input_binmask == i]
                         #if at least 10 peaks in the bin
                         bin_resmean = np.nanmean(res_bin)
                         #get std from entire population
                         if population == "all":
                              col_res.append(bin_resmean)
                              col_std.append(np.nanstd(input_data_bin))
                              col_tmean.append(np.nanmean(input_data_bin))
                         #mean of obsss
                         if population == "obss":
                              col_smean.append(np.nanmean(input_data_bin))
                              col_sstd.append(np.nanstd(input_data_bin))
                         #mean of obsw
                         if population == "obsw":
                              col_wmean.append(np.nanmean(input_data_bin))
                              col_wstd.append(np.nanstd(input_data_bin))
               #convert lists to np array
               binstd = np.array(col_std)
               binmean = np.array(col_tmean)
               binres = np.array(col_res)
               mean_sval = np.array(col_smean)
               std_sval = np.array(col_sstd)
               mean_wval = np.array(col_wmean)
               std_wval = np.array(col_wstd)
               
               print "     FITTING RESOLUTION DEPENDENT SPLINE FUNCTIONS",column

               # after PCA,take bin mean as average of s and w populations (centers data)
               if post_pca:
                    binmean = np.divide(mean_wval + mean_sval,2.0)
               mean_spline_coeff = self.spline4k_fit(binres,binmean,augment=True)
               #sig must be positive, fit log sigma
               log_sig_spline_coeff = self.spline4k_fit(binres,np.log(binstd),augment=True)
               scales_dict[column] = (tuple(mean_spline_coeff),tuple(log_sig_spline_coeff))

               #plotting functions to test fit
               if plot:
                    sub = gridplot.add_subplot(4,5,index+1)
                    plot_xval = np.linspace(0.5,5.0,100)
                    mean_plot_spline = self.spline4k(mean_spline_coeff,plot_xval)
                    sig_plot_spline = np.exp(self.spline4k(log_sig_spline_coeff,plot_xval))
                    sub.text(0.9,0.9,"%s" % column,verticalalignment='bottom',horizontalalignment='right',
                             transform=sub.transAxes,fontsize=12)
                    sub.set_xlim([0.5,5.0])
                    sub.plot(plot_xval,mean_plot_spline,'k-')
                    sub.plot(plot_xval,sig_plot_spline,'c-')  
                    sub.scatter(binres,mean_wval,color='blue')
                    sub.scatter(binres,mean_sval,color='red')
                    sub.scatter(binres,binmean,color='black')
                    sub.scatter(binres,binstd,color='cyan')
                    ax2=sub.twinx()
                    sn=np.divide((np.power(np.subtract(mean_sval,mean_wval),2)),binstd**2)
                    ax2.set_ylim([0.0,np.clip(np.amax(sn),2.0,20.0)])
                    ax2.plot(binres,sn,color='green')
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
          Calculates resolution dependent PCA modal matrices binwise
          1) calculate a reference modal using all data
          2) calculate modal matricies by resolution bin
          3) fit outputs vs. resolution by splines
          """
          print "CALCULATING RESOLUTION DEPENDENT PCA MATRIX COEFFICIENTS"
          selectors = self.ppsel(norm_data)
          res = norm_data['res']
          col_list = selectors.std_view_col
          view_dtype = selectors.std_view_dtype
          #select numerical data
          selected_data = norm_data[col_list].view(selectors.raw_dtype)
          total_modal = self.modal_matrix(selected_data,verbose=True)

          #calculte PCA components in chunks by resolution, or by size/bin
          #generate binw a minimum number of sulfate and a target minimum resolution width
          bin_mask,num_bins = self.calc_res_bins(norm_data,20,target_res_bin_width=0.04)
          eig_data,binres_vals = self.pca_by_bin(selected_data,res,bin_mask,num_bins,plot=True)

          #initialize empty 3d array for sorted/oriented modal matrices
          oriented_pca = np.zeros(eig_data.shape)

          #iterate by resolution bin (first index)
          for pcabin,bin_modal in enumerate(eig_data):
               #reference is initially the PCA xfrom of entire dataset
               if pcabin == 0:
                    ref_modal = total_modal
               new_modal = self.eig_sort(ref_modal,bin_modal,verbose=True)
               oriented_pca[pcabin] = new_modal
               #once first bin set, reference is previous window to enforce smoothing
               ref_modal = new_modal


          #fit the resolution dependent PCA components to a usual spline function
          #initialize output, shape of modal with 6 coefficients for each element
          pca_coeffs = np.zeros((oriented_pca.shape[1],oriented_pca.shape[2],6))
          for modal_j in np.arange(oriented_pca.shape[2]):
               print "     FITTING EIGv %2s COMPONENTS" % modal_j
               if plot:
                    gridplot = plt.figure(figsize=(24,8))
               for modal_i in np.arange(oriented_pca.shape[1]):
                    xval = binres_vals
                    #slice by first index (resolution)
                    yval = np.array(oriented_pca[:,modal_i,modal_j])
                    pca_coeffs[modal_i,modal_j,:] = self.spline4k_fit(xval,yval,augment=True)
                    #reject single worst outlier, testing training  
                    residuals = (yval - self.spline4k(pca_coeffs[modal_i,modal_j,:],xval))**2
                    exclude_worst_point = np.argsort(residuals)[0:-1]
                    #reject bad points (testing/training)
                    xval = xval[exclude_worst_point]
                    yval = yval[exclude_worst_point]
                    pca_coeffs[modal_i,modal_j,:] = self.spline4k_fit(xval,yval,augment=True)

                    #plotting functions
                    if plot:
                         fitxval = np.linspace(0.5,5.0,100)
                         fityval = self.spline4k(pca_coeffs[modal_i,modal_j,:],fitxval)
                         sub = gridplot.add_subplot(4,5,modal_i+1)
                         sub.text(0.1,0.8,"%s" % modal_i,verticalalignment='bottom',horizontalalignment='right',
                                  transform=sub.transAxes,fontsize=12)
                         sub.set_xlim([0.5,5.0])
                         maxval=np.amax(np.absolute(yval))
                         plot_ylim = np.clip(maxval,1.0,np.inf)
                         sub.set_ylim([-plot_ylim,plot_ylim])
                         sub.scatter(xval,yval)
                         sub.plot(fitxval,fityval)
                         #add line to indicate total pca value
                         sub.plot((0.5,5.0),(total_modal[modal_i,modal_j],total_modal[modal_i,modal_j]))
               if plot:
                    plt.savefig("PCA_eigv_reject_fits_"+str(modal_j)+".png")
                    plt.clf()
                    plt.close()

          #store coeffs in a dictionary and write
          pca_coeffs_dict = {}
          for i in range(pca_coeffs.shape[0]):
               for j in range(pca_coeffs.shape[1]):
                    pca_coeffs_dict[str(i)+"_"+str(j)] = tuple(pca_coeffs[i,j])
          f = open("pprobe_pca_matrix_coeff.dict",'w')
          print "WRITING PCA COEFFICIENTS"
          f.write(str(pca_coeffs_dict))
          f.close()


     def check_decorr(self,norm_data,pca_data,plot=False):
          print "CHECKING DECORRELATIONS"
          selectors = self.ppsel(norm_data)
          res = pca_data['res']
          n_col_list = selectors.std_view_col
          p_col_list = selectors.pca_view_col
          #select numerical data
          selected_norm_data = norm_data[n_col_list].view(selectors.raw_dtype)
          selected_pca_data = pca_data[p_col_list].view(selectors.raw_dtype)
          bin_mask,num_bins = self.calc_res_bins(pca_data,20,target_res_bin_width=0.04)
          for resbin in np.arange(num_bins + 1):
               ressel = np.logical_and(bin_mask >= resbin,bin_mask <= resbin)
               n_databin = selected_norm_data[ressel]
               p_databin = selected_pca_data[ressel]
               res_block=res[ressel]
               calc_modal = self.gen_xform_mat(np.nanmean(res_block))
               #untransformed data transformed locally
               block_n_corr = np.corrcoef(n_databin.T)
               block_modal,l,v = np.linalg.svd(block_n_corr)
               block_modal_sort = self.eig_sort(calc_modal,block_modal)
               block_n_xform = np.dot(n_databin,block_modal)
               block_np_corr = np.corrcoef(block_n_xform.T)
               block_ns_xform = np.dot(n_databin,block_modal_sort)
               block_ns_corr = np.corrcoef(block_ns_xform.T)
               #data transformed by fitted modal matrix

               block_p_corr_asis = np.corrcoef(p_databin.T)
               rx_p = np.dot(n_databin,calc_modal)
               rx_pcorr = np.corrcoef(rx_p.T)


               print "DECORR FOR BIN",resbin,res_block[0],res_block[-1]
               print "INPUT CORR"
               print block_n_corr
               print "INPUT XCORR ASIS"
               print block_np_corr
               print "INPUT XCORR SORT"
               print block_ns_corr
               print "OUTPUT CORR ASIS"
               print block_p_corr_asis
               print "OUTPUT CORR RECALC"
               print rx_pcorr
               print "FITTED MODAL"
               print calc_modal
               print "LOCAL MODAL"
               print block_modal_sort
               print "NORM F-L",np.linalg.norm(np.subtract(calc_modal,block_modal_sort))
               q,r = np.linalg.qr(calc_modal)
               print "QR-R NORM SCALED",np.linalg.norm(r) - np.sqrt(len(n_col_list))


     def calc_res_bins(self,data_array,min_so4_per_bin,target_res_bin_width=0.075):
          #bins data, returns integer array mask
          #input data must be sorted by resolution
          #if fastmode set, binsizes increased for faster calculations
          if self.fastmode:
               min_so4_per_bin = min_so4_per_bin*5
               target_res_bin_width = target_res_bin_width*5 
          selectors = self.ppsel(data_array)
          array_size = data_array.shape[0]
          bin_mask=np.zeros(array_size,dtype=np.int16)
          binno = 0
          scount = 0
          tot_count = 0
          reswidth = 0.0
          binres_max = data_array['res'][-1]
          binres_min = 0.0
          #start from the high(low) res, change bin number
          #once min_so4 counts is reached and target bin width reached
          for index,ori in enumerate(data_array['ori'][::-1]):
               bin_mask[index] = binno
               tot_count = tot_count + 1
               if ori == "SO4" or ori == "PO4":
                    scount = scount + 1
                    binres_min = data_array['res'][-index]
                    reswidth = binres_max - binres_min
               if scount >= min_so4_per_bin and reswidth >= target_res_bin_width:
                    #print "BIN FULL",binno,scount,tot_count,binres_max,binres_min,reswidth
                    scount = 0
                    binno = binno + 1
                    binres_max = data_array['res'][-index]
               if index == array_size -1:#hit the end of the array
                    #group last two (highest res) bins together
                    binno = binno -1
                    bin_mask[bin_mask == binno] = binno
                    #print "LAST BIN ADDED TO PREVIOUS",binno,scount,tot_count,binres_max,binres_min,reswidth

          #flip array as we started at the end, then flip bin numbers from best res to worst
          bin_mask = np.flipud(bin_mask)
          bin_mask = binno - bin_mask
          print "DATA BINNING"
          for i in range(binno + 1):
               binsize = np.count_nonzero(bin_mask == i)
               no_so4 = np.count_nonzero(np.logical_and(selectors.inc_obss_bool,bin_mask==i))
               binres = np.nanmean(data_array['res'][bin_mask == i])
               print "    BIN %3s SIZE %7s SO4 %6s RES %4.2f" % (i,binsize,no_so4,binres)
          return bin_mask,binno

     def calc_jsu_coeff(self,data_array,plot=False):
          """
          takes data array with flagged populations (sulfate/water)
          generates a histogram of each population and a resolution bin, 
          then fits a johnsonsu distribution to each population
          then calculates a function to fit the coefficients of the distribution
          as a function of resolution
          """
          print "FITTING TRANSFORMED DATA TO Jsu DISTRIBUTIONS"
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
          #divide resolution into bins
          bin_mask,num_bins = self.calc_res_bins(data_array,50,target_res_bin_width=0.075)
          swinit = [[],[]] #initialize restraints empty list of two lists
          for resbin in np.arange(num_bins + 1):
               #using moving window of three bins
               ressel = np.logical_and(bin_mask >= resbin,bin_mask < resbin+3)
               selected_data = data_array[ressel]
               print "     FITTING JSU HISTOGRAM FOR BIN %.2f %.2f" % (selected_data['res'][0],
                                                                       selected_data['res'][-1])
               if resbin == 0:
                    sfit_list,wfit_list,res_list = self.pop_fit_jsu(selected_data,plot=plot)
                    swinit[0],swinit[1] = sfit_list,wfit_list #store fitted values
                    print "INITIAL FIT RX0",swinit[0][0],swinit[1][0]
               else:
                    sfit_list,wfit_list,res_list = self.pop_fit_jsu(selected_data,swivals=swinit,
                                                                    restrain=True,plot=plot)
                    swinit[0],swinit[1] = sfit_list,wfit_list #updated fitted values
               res_data.append(res_list)
               sfit_data.append(sfit_list)
               wfit_data.append(wfit_list)
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
               sub.set_ylim([0.0,np.clip(np.amax(sn),2.0,np.inf)])
               sub.plot(xplot,sn)
               sub.set_title("SN_"+str(index))
          sub=gridplot.add_subplot(4,5,20)
          #total SN as addition?
          sub.plot(xplot,sn_sum)
          gridplot.savefig("sn_plot.png")
          plt.close()

     def pop_fit_jsu(self,data_array,swivals=None,restrain=False,plot=False):
          #fits a batch of data containing both so4 and water
          #to JSU distributions, is passed one "bin" of data by resolution
          #use restrained fitting or initial values based on calling routine
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
          no_so4 = np.count_nonzero(obss_sel)
          no_wat = np.count_nonzero(obsw_sel)

          for index,column in enumerate(col_list):
               input_column=data_array[column]
               #set sensible histogram limits
               slow = np.percentile(input_column[obss_sel],0.1)
               wlow = np.percentile(input_column[obsw_sel],0.1)
               shigh = np.percentile(input_column[obss_sel],99.9)
               whigh = np.percentile(input_column[obsw_sel],99.9)
               #old formula for calculating binwidth
               sbinw = np.nanstd(input_column[obss_sel])*np.power(42.5/no_so4,0.33)
               wbinw = np.nanstd(input_column[obsw_sel])*np.power(42.5/no_wat,0.33)
               #divide range by binwidth, clip to reasonable number of points
               sbins = np.clip(int((shigh-slow)/sbinw),20,100)
               wbins = np.clip(int((whigh-wlow)/wbinw),20,100)
               obss_hist,obss_bins = np.histogram(input_column[obss_sel],bins=sbins,density=True,range=(slow,shigh))
               obsw_hist,obsw_bins = np.histogram(input_column[obsw_sel],bins=wbins,density=True,range=(wlow,whigh))
               #calculates average of bin edges for fitting
               if swivals:
                    if restrain:
                         sfit = self.fit_jsu(((obss_bins[:-1] + obss_bins[1:]) / 2.0),
                                             obss_hist,ivals=swivals[0][index],restrain=True)
                         wfit = self.fit_jsu(((obsw_bins[:-1] + obsw_bins[1:]) / 2.0),
                                             obsw_hist,ivals=swivals[1][index],restrain=True)
                    else:
                         sfit = self.fit_jsu(((obss_bins[:-1] + obss_bins[1:]) / 2.0),
                                             obss_hist,ivals=swivals[0][index],restrain=False)
                         wfit = self.fit_jsu(((obsw_bins[:-1] + obsw_bins[1:]) / 2.0),
                                             obsw_hist,ivals=swivals[1][index],restrain=False)
               else:
                    sfit = self.fit_jsu(((obss_bins[:-1] + obss_bins[1:]) / 2.0),obss_hist)
                    wfit = self.fit_jsu(((obsw_bins[:-1] + obsw_bins[1:]) / 2.0),obsw_hist)

               sfit_list.append(sfit)
               wfit_list.append(wfit)
               res_list.append(np.nanmean(data_array['res']))

               if plot: 
                    xdata=np.linspace(np.amin((slow,wlow))-1.0,np.amax((shigh,whigh))+1.0,100)
                    sub = gridplot.add_subplot(4,5,index+1)
                    splot_bins = obss_hist.shape[0]
                    wplot_bins = obsw_hist.shape[0]
                    sub.plot(xdata,self.johnsonsu_pdf(xdata,sfit[0],sfit[1],sfit[2],sfit[3]),'r-')
                    sub.plot(xdata,self.johnsonsu_pdf(xdata,wfit[0],wfit[1],wfit[2],wfit[3]),'b-')
                    sub.hist(input_column[obss_sel], normed=True, bins=splot_bins,range=(slow,shigh),color="red",alpha=0.5)
                    sub.hist(input_column[obsw_sel], normed=True, bins=wplot_bins,range=(wlow,whigh),color="blue",alpha=0.5,)
                    sub.text(0.2,0.8,"%s" % column,verticalalignment='bottom',horizontalalignment='right',
                             transform=sub.transAxes,fontsize=12)
                    sstat = self.johnsonsu_stats(sfit)
                    wstat = self.johnsonsu_stats(wfit)
                    sub.text(0.95,0.85,"%.1f %.2f" % (sstat[0],np.sqrt(sstat[1])),verticalalignment='bottom',
                             horizontalalignment='right',transform=sub.transAxes,fontsize=12,color='red')
                    sub.text(0.95,0.75,"%.1f %.2f" % (wstat[0],np.sqrt(wstat[1])),verticalalignment='bottom',
                             horizontalalignment='right',transform=sub.transAxes,fontsize=12,color='blue')
                    sub.text(0.95,0.65,"%.2f %.2f %.2f %.2f" % tuple(sfit),verticalalignment='bottom',
                             horizontalalignment='right',transform=sub.transAxes,fontsize=10,color='red')
                    sub.text(0.95,0.55,"%.2f %.2f %.2f %.2f" % tuple(wfit),verticalalignment='bottom',
                             horizontalalignment='right',transform=sub.transAxes,fontsize=10,color='blue')
          if plot:
               plt_str = str(res_list[-1])[0:4]
               plt.savefig("JSU_FITS_"+plt_str+".png")
               plt.clf()
               plt.close()
          return sfit_list,wfit_list,res_list     
                           
          
     def population_dist_stats(self,data_array):
          selectors = self.ppsel(data_array)
          gridplot = plt.figure(figsize=(24,16))
          res = data_array['res']
          col_names=selectors.pca_view_col
          res_list,smean_list,wmean_list,sstd_list,wstd_list = [],[],[],[],[]
          jsu_smean_list,jsu_sstd_list,jsu_wmean_list,jsu_wstd_list = [],[],[],[]
          bin_mask,num_bins = self.calc_res_bins(data_array,50,target_res_bin_width=0.075)
          for index,column in enumerate(col_names):         
               res_list,smean_list,wmean_list,sstd_list,wstd_list = [],[],[],[],[]
               jsu_smean_list,jsu_sstd_list,jsu_wmean_list,jsu_wstd_list = [],[],[],[]
               input_data = data_array[column]
               for i in np.arange(num_bins + 1):
                    input_data_bin=input_data[bin_mask == i]
                    ssel = selectors.inc_obss_bool[bin_mask==i]
                    wsel = selectors.inc_obsw_bool[bin_mask==i]
                    res_bin=res[bin_mask == i]
                    bin_resmean = np.nanmean(res_bin)
                    res_list.append(bin_resmean)
                    smean_list.append(np.nanmean(input_data_bin[ssel]))
                    wmean_list.append(np.nanmean(input_data_bin[wsel]))
                    sstd_list.append(np.nanstd(input_data_bin[ssel]))
                    wstd_list.append(np.nanstd(input_data_bin[wsel]))
                    s_pdf_coeff,w_pdf_coeff=self.get_jsu_coeffs(index,bin_resmean)
                    sstats = self.johnsonsu_stats(s_pdf_coeff)
                    wstats = self.johnsonsu_stats(w_pdf_coeff)
                    jsu_smean_list.append(sstats[0])
                    jsu_sstd_list.append(np.sqrt(sstats[1]))
                    jsu_wmean_list.append(wstats[0])
                    jsu_wstd_list.append(np.sqrt(wstats[1]))
               sub = gridplot.add_subplot(4,5,index+1)
               xval = np.array(res_list)
               smean = np.array(smean_list)
               wmean = np.array(wmean_list)
               sstd = np.array(sstd_list)
               wstd = np.array(wstd_list)
               sjmean = np.array(jsu_smean_list)
               wjmean = np.array(jsu_wmean_list)
               sjstd = np.array(jsu_sstd_list)
               wjstd = np.array(jsu_wstd_list)
               sub.plot(xval,smean,'r-')
               sub.plot(xval,wmean,'b-')
               sub.plot(xval,sstd,'m-')
               sub.plot(xval,wstd,'c-')
               sub.scatter(xval,sjmean,color='red',alpha=0.5)
               sub.scatter(xval,wjmean,color='blue',alpha=0.5)
               sub.scatter(xval,sjstd,color='magenta',alpha=0.5)
               sub.scatter(xval,wjstd,color='cyan',alpha=0.5)
               total_means = [np.nanmean(x) for x in (smean,sjmean,sstd,sjstd,wmean,wjmean,wstd,wjstd)]
               print "MEANS COLUMN S jS sS jsS W jW sW sjW",column,list('{:.2f}'.format(x) for x in total_means)
               

          plt.savefig("compare_stats.png")
          plt.clf()
          plt.close()

               
