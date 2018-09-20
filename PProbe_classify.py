import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
from PProbe_selectors import Selectors
from PProbe_stats import StatFunc
from PProbe_util import Util
from PProbe_matrix import PCA
from PProbe_dataio import DataIO
from PProbe_filter import Filters
from PProbe_contacts import Contacts
from PProbe_util import Util
#for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

class ClassifierFunctions:
     """
     uses the some of the same functions as the FitFunctions class in PPutil, but not the fitting functions
     or anything that requires scipy, whose import conflicts with things in phenix.python
     """
     def __init__(self,input_dict=None,input_dfile=None,verbose=False,train=False):
          self.verbose = verbose
          self.ppsel = Selectors
          self.ppstat = StatFunc()
          self.pppca = PCA()
          self.ppio = DataIO()
          self.pput = Util()
          self.ppcont = Contacts()
          self.ppfilt = Filters(verbose=verbose)
          self.johnsonsu_stats = self.ppstat.johnsonsu_stats
          self.johnsonsu_pdf = self.ppstat.johnsonsu_pdf
          self.norm_pdf = self.ppstat.norm_pdf
          self.llr_to_prob = self.ppstat.llr_to_prob
          self.spine_basis = self.ppstat.spline_basis
          self.spline4k = self.ppstat.spline4k
          self.fishers_disc = self.ppstat.fishers_disc
          self.resscale_pca_coeffs = None
          self.scale_post_pca_spline_coeffs = None
          self.scale_raw_feature_spline_coeffs = None
          self.contact_coeffs = None
          self.jsu_coeffs = None
          self.scale_composite = None
          self.chiD_coeffs = None
          if not train:
               self.setup_dict(input_dict=input_dict,input_dfile=input_dfile)


     def setup_dict(self,input_dict=None,input_dfile=None):
          #reads in model param dictionary and sets up coefficient dictionaries for all scaling/processing
          if input_dict is None and input_dfile is None:
               #try reading default "pprobe_master.dict"
               self.master_dict = self.ppio.read_master_dict()
          elif input_dict is None:
               self.master_dict = self.ppio.read_master_dict(input_dfile=input_dfile)
          else:
               self.master_dict = input_dict
          cur_keys = self.master_dict.keys()
          if 'respost' in cur_keys:
               self.scale_post_pca_spline_coeffs = self.master_dict['respost']
          if 'respre' in cur_keys:
               self.scale_raw_feature_spline_coeffs = self.master_dict['respre']
          if 'contacts' in cur_keys:
               self.contact_coeffs = self.master_dict['contacts']
          if 'density' in cur_keys:
               self.jsu_coeffs = self.master_dict['density']
          if 'pca' in cur_keys:
               self.resscale_pca_coeffs = self.master_dict['pca']
          if 'composite' in cur_keys:
               self.scale_composite = self.master_dict['composite']
          if 'chiD' in cur_keys:
               self.chiD_coeffs = self.master_dict['chiD']


     def get_stats(self,array):
          if array.shape[0] == 0:
               mean,stdev = 0.0,1.0
          else:
               mean,stdev = np.nanmean(array,dtype=np.float64),np.nanstd(array,dtype=np.float64)
          if stdev == 0: #avoid divide by zero for only one point?  
               stdev = 1.0
          return mean,stdev

     def get_res_scales(self,column_name,resolution):
     #takes a column name and a resolution value (or 1d array)
     #returns the scaling factors based on resolution
     #mean is the average of sulfate/phosphate and water populations
     #sig is the total standard deviation
          mean_scale = self.spline4k(self.scale_raw_feature_spline_coeffs[column_name][0],resolution)
          sig_scale = self.spline4k(self.scale_raw_feature_spline_coeffs[column_name][1],resolution)
          return mean_scale,sig_scale

     def get_post_pca_scales(self,column_name,resolution):
     #similar to above, but coefficients are for data following PCA transformation
     #Rxx = transformed by pca from resolution dependent covariance
          mean_scale = self.spline4k(self.scale_post_pca_spline_coeffs[column_name][0],resolution)
          sig_scale = self.spline4k(self.scale_post_pca_spline_coeffs[column_name][1],resolution)
          return mean_scale,sig_scale

     def get_composite_scales(self,column_name,resolution):
     #similar to above, but coefficients are for score,cscore
          mean_scale = self.spline4k(self.scale_composite[column_name][0],resolution)
          sig_scale = self.spline4k(self.scale_composite[column_name][1],resolution)
          return mean_scale,sig_scale

     def get_jsu_coeffs(self,column_num,resolution):
          if self.jsu_coeffs is None:
               sys.exit("NO JSU COEFFICIENTS, ERROR!")
          #hackish function to retreive coefficients from dictionary above and 
          #return the 4 coeffs for the jsu pdf based on resolution
          #can be passed a single value, or an array
          column_num = str(column_num)
          scol_c1 = self.jsu_coeffs["SC"+column_num+"_jsuc1"]
          scol_c2 = self.jsu_coeffs["SC"+column_num+"_jsuc2"]
          scol_c3 = self.jsu_coeffs["SC"+column_num+"_jsuc3"]
          scol_c4 = self.jsu_coeffs["SC"+column_num+"_jsuc4"]
          wcol_c1 = self.jsu_coeffs["WC"+column_num+"_jsuc1"]
          wcol_c2 = self.jsu_coeffs["WC"+column_num+"_jsuc2"]
          wcol_c3 = self.jsu_coeffs["WC"+column_num+"_jsuc3"]
          wcol_c4 = self.jsu_coeffs["WC"+column_num+"_jsuc4"]
          s_pdf_a = self.spline4k(scol_c1,resolution)
          s_pdf_b = self.spline4k(scol_c2,resolution)
          s_pdf_loc = self.spline4k(scol_c3,resolution)
          s_pdf_scale = self.spline4k(scol_c4,resolution)
          w_pdf_a = self.spline4k(wcol_c1,resolution)
          w_pdf_b = self.spline4k(wcol_c2,resolution)
          w_pdf_loc = self.spline4k(wcol_c3,resolution)
          w_pdf_scale = self.spline4k(wcol_c4,resolution)
          s_pdf_coeff=np.array((s_pdf_a,s_pdf_b,s_pdf_loc,s_pdf_scale))
          w_pdf_coeff=np.array((w_pdf_a,w_pdf_b,w_pdf_loc,w_pdf_scale))
          return s_pdf_coeff,w_pdf_coeff

     def gen_xform_mat(self,res):
          #takes a resolution and outputs a pseudo modal matrix generated from spline
          #coefficients, works "one at a time" 
          if self.resscale_pca_coeffs is None:
               sys.exit("NO PCA COEFFICIENTS, ERROR!")
          pca_coeffs = self.resscale_pca_coeffs
          num_col = int(np.sqrt(len(pca_coeffs)))
          #initialze a matrix
          xform_matrix = np.zeros((num_col,num_col),dtype=np.float64)
          for i in range(xform_matrix.shape[0]):
               for j in range(xform_matrix.shape[1]):
                    column_index=str(i)+"_"+str(j)
                    params = pca_coeffs[column_index]
                    xform_matrix[i,j] = self.spline4k(params,res.reshape(-1))
          #these regenerated modal matrices are often have det != 1.0 and introduce
          #strange artifacts into the data
          #the following trick "squares up" the matrix
          modal_u,modal_l,modal_v = np.linalg.svd(xform_matrix)
          modal_fixed = np.dot(modal_u,modal_v)
          orig_det = np.linalg.det(xform_matrix)
          fixed_det = np.linalg.det(modal_fixed)
          eval_sum = np.nansum(modal_l,dtype=np.float64) #should be tr(M), but isn't sometimes
          if self.verbose:
               print "     PCA MODAL MATRIX FOR RESOLUTION %.2f DET %.3f/%.3f" % (res,orig_det,fixed_det)
          return modal_fixed


     def xform_data(self,data,matrix):
          #data as rows
          #matrix is modal (eigenbasis as columns)
          xformed_data = np.dot(data,matrix)
          return xformed_data

     def check_xform(self,data,matrix,verbose=False):
          #pass only numerical data
          #check a batch of data against a single modal matrix
          #matrix is modal (eigenbasis as columns)
          if data.shape[0] > 20:
               xformed_data = np.dot(data,matrix)
               print "INPUT COV"
               print np.cov(data.T)
               print "OUTPUT COV"
               print np.cov(xformed_data.T)
          else:
               print "INSUFFICIENT DATA FOR COV ESTIMATION"



     def standardize_data(self,data_array,post_pca=False,composite=False):
          """
          normalize/standardize in place (subtract mean, divide by std)
          centers data by subtracting the resolution dependent s/w mean and sigma
          centering puts zero as the midpoint between sulfate and water means
          """
          if self.verbose:
               print "STANDARDIZING DATA:"
          selectors=Selectors(data_array)
          #read in the dictionary of spline coefficients for raw data scaling

          res = data_array['res']
          resolution_column=np.clip(res,0.6,5.0).astype(np.float16)
          #method works with data both before and after PCA transformation
          #setup accordingly with array column names and scale coeff dictionaries
          if post_pca:
               if composite:
                    column_list = ['score','cscore']
                    if self.scale_composite is None:
                         sys.exit("NO COMPOSITE COEFFICIENTS, ERROR!")
               else:
                    column_list = selectors.pca_view_col
                    if self.scale_post_pca_spline_coeffs is None:
                         sys.exit("NO POSTRES COEFFICIENTS, ERROR!")

          else:
               column_list = selectors.std_view_col
               if self.scale_raw_feature_spline_coeffs is None:
                    sys.exit("NO PRERES COEFFICIENTS, ERROR!")

          for column in column_list:
               calc_meanx,calc_stdev = self.get_stats(data_array[column])
               if self.verbose:
                    print "      INPUT COLUMN %10s MEAN %4.2f SIG %4.2f" % (column,calc_meanx,calc_stdev)
               if post_pca == False:
                    resd_mean,resd_sig = self.get_res_scales(column,resolution_column)
               else:
                    if composite:
                         resd_mean,resd_sig = self.get_composite_scales(column,resolution_column)
                    else:
                         resd_mean,resd_sig = self.get_post_pca_scales(column,resolution_column)
               data_array[column] = np.divide(np.subtract(data_array[column],resd_mean),np.exp(resd_sig))
               #check normalization applied correctly
               calc_meanx,calc_stdev = self.get_stats(data_array[column])
               if self.verbose:
                    print "     OUTPUT COLUMN %10s MEAN %4.2f SIG %4.2f" % (column,calc_meanx,calc_stdev)
          if self.verbose:
               print "SCALED %s ROWS DATA" % data_array.shape[0]

     def pca_xform_data(self,data_array):
          """
          carries out PCA transformation, first generates matrix for PCA transformation given
          an input resolution, then applies it to one or more data points
          data must be sorted by resolution
          """
          if self.verbose:
               print "PCA TRANSFORMATION:"
          selectors=Selectors(data_array)
          num_data = data_array[selectors.std_view_col].view(selectors.raw_dtype)
          res = data_array['res']
          xformed_data = np.zeros(num_data.shape,dtype=np.float64)
          #read in PCA transformation matrix coefficients
          #each entry is an i,j matrix entry with 6 spline coefficients used to calculate the eigenvector component at input resolution          
          if self.resscale_pca_coeffs is None:
               sys.exit("NO PCA COEFFICIENTS, ERROR!")
          #fetch a matrix for the 1st peak
          cur_res = res[0]
          mat = self.gen_xform_mat(res[0])
          for index,row in enumerate(num_data):
               #update matrix if resolution is more than 0.02A different (speedup)
               #matrix coefficients are smoothly varying
               if np.abs(res[index] - cur_res) > 0.02:
                    mat = self.gen_xform_mat(res[index])
                    cur_res = res[index]
               xformed_data[index] = self.xform_data(row,mat)
          #put everything back in the data processing array
          for index,col in enumerate(selectors.pca_view_col):
               data_array[col] = xformed_data[:,index].astype(np.float32)
          array_size = num_data.shape[0]
          if self.verbose:
               print "TRANSFORMED %s ROWS DATA" % array_size


     def density_da(self,data_array,master_array):
          """
          Akin to discriminant analysis, calculates relative likelihoods that a peak comes
          about from random normal data, then from one population or another
          also calculates a chi-sq statistic from jsu norm/sigma (not really a chi-sq)
          """
          if self.verbose:
               print "SCORING %s PEAKS ON DENSITY FEATURES" % data_array.shape[0]
          selectors = Selectors(master_array)
          pput = self.pput
          if self.jsu_coeffs is None:
               sys.exit("NO JSU COEFFICIENTS, ERROR!")
          #can use large amounts of memory, break into sub arrays if necessary
          subind_arr = pput.batch_data_equal(master_array,100000)
          n_subarray = np.amax(subind_arr) + 1
          if n_subarray > 1:
               print "  Using %s subarrays" % n_subarray
          for sub_ind in range(n_subarray):
               #initialize arrays
               #for each feature, likelihood values from jsu distributions
               sub_array = data_array[subind_arr == sub_ind]
               sub_res = sub_array['res']
               likelihood_s = np.zeros((sub_array.shape[0],len(selectors.pca_view_col)),dtype=np.float64)
               likelihood_w = np.zeros(likelihood_s.shape,dtype=np.float64)
               #baselines are for calibration of LLG, data is centered with the average
               #between s and w populations set to zero, so inputting zero as observations
               #estimates baseline likelihood given 50/50 probability
               baseline_rand = np.zeros(likelihood_s.shape,dtype=np.float64)
               dev_s = np.zeros(likelihood_s.shape,dtype=np.float64)
               dev_w = np.zeros(likelihood_s.shape,dtype=np.float64)
               jsu_mean_s = np.zeros(likelihood_s.shape,dtype=np.float64)
               jsu_mean_w = np.zeros(likelihood_s.shape,dtype=np.float64)
               jsu_var_s = np.zeros(likelihood_s.shape,dtype=np.float64)
               jsu_var_w = np.zeros(likelihood_s.shape,dtype=np.float64)
               #iterate by feature (column) -- bit clumsy
               for index,column in enumerate(selectors.pca_view_col):
                    s_pdf_coeff,w_pdf_coeff=self.get_jsu_coeffs(index,sub_res)
                    likelihood_s[:,index] = self.johnsonsu_pdf(sub_array[column],*s_pdf_coeff)
                    likelihood_w[:,index] = self.johnsonsu_pdf(sub_array[column],*w_pdf_coeff)
                    jsu_mean_s[:,index],jsu_var_s[:,index] = self.johnsonsu_stats(s_pdf_coeff)
                    jsu_mean_w[:,index],jsu_var_w[:,index] = self.johnsonsu_stats(w_pdf_coeff)
                    baseline_rand[:,index] = self.norm_pdf(sub_array[column])
                    #store deviations from distribution means
                    dev_s[:,index] = np.subtract(sub_array[column],jsu_mean_s[:,index])
                    dev_w[:,index] = np.subtract(sub_array[column],jsu_mean_w[:,index])


               #clip likelihoods to avoid underrun and artifacts from imperfect distributions (in place)
               np.clip(likelihood_s,0.0001,np.inf,out=likelihood_s)
               np.clip(likelihood_w,0.0001,np.inf,out=likelihood_w)
               np.clip(jsu_var_s,0.001,np.inf,out=jsu_var_s)
               np.clip(jsu_var_w,0.001,np.inf,out=jsu_var_w)
               np.clip(baseline_rand,0.00001,np.inf,out=baseline_rand)
               #linear ind likelihoods, sum logs for total
               ll_s = np.nansum(np.log(likelihood_s),dtype=np.float64,axis=1)
               ll_w = np.nansum(np.log(likelihood_w),dtype=np.float64,axis=1)
               ll_rand = np.nansum(np.log(baseline_rand),axis=1,dtype=np.float64)
               #added to give log likelihood gain LLG
               llg_s = ll_s - ll_rand
               llg_w = ll_w - ll_rand
               llg_ratio = np.subtract(llg_s,llg_w)
               #chisq calculations, inline multiplication faster
               chisq_s = np.nansum(np.divide(np.multiply(dev_s,dev_s),jsu_var_s),axis=1,dtype=np.float64)
               chisq_w = np.nansum(np.divide(np.multiply(dev_w,dev_w),jsu_var_w),axis=1,dtype=np.float64)
               #store score as LLG ratio 
               #write to pre-instantiated structured array

               master_array['score'][subind_arr == sub_ind] = llg_ratio.astype(np.float32)
               master_array['llgS'][subind_arr == sub_ind] = llg_s.astype(np.float32)
               master_array['llgW'][subind_arr == sub_ind] = llg_w.astype(np.float32)
               master_array['chiS'][subind_arr == sub_ind] = chisq_s.astype(np.float32)
               master_array['chiW'][subind_arr == sub_ind] = chisq_w.astype(np.float32)


     def contact_da(self,data_array):
          """
          distance to first contact (heavy) and local environment contact ('charge')
          are compared to jsu distributions to get a likelihood ratio
          data are normalized, all cofficients come from training stored in 
          dictionary "pprobe_contact_coeffs.dict"
          """
          if self.verbose:
               print "SCORING %s PEAKS ON CONTACT FEATURES" % data_array.shape[0]
          if self.contact_coeffs is None:
               sys.exit("NO CONTACT COEFFICIENTS, ERROR!")
          coeff_dict = self.contact_coeffs
          column_list = ['charge','c1']
          norm_data = np.zeros((data_array.shape[0],len(column_list)))
          for index,column in enumerate(column_list):
               norm_data[:,index] = np.divide(np.subtract(data_array[column],coeff_dict['mean_'+column]),coeff_dict['std_'+column])
          likelihood_s = np.zeros((data_array.shape[0],len(column_list)),dtype=np.float64)
          likelihood_w = np.zeros(likelihood_s.shape,dtype=np.float64)
          baseline_rand = np.zeros(likelihood_s.shape,dtype=np.float64)
          dev_s = np.zeros(likelihood_s.shape,dtype=np.float64)
          dev_w = np.zeros(likelihood_s.shape,dtype=np.float64)
          jsu_mean_s = np.zeros(likelihood_s.shape,dtype=np.float64)
          jsu_mean_w = np.zeros(likelihood_s.shape,dtype=np.float64)
          jsu_var_s = np.zeros(likelihood_s.shape,dtype=np.float64)
          jsu_var_w = np.zeros(likelihood_s.shape,dtype=np.float64)
          #based on density_da above, but modified
          #with only two features input, output ratios are peaky and start to become discrete
          #so here, likelihoods are multiplied by z_scores with means/stds taken
          #from JSU coefficients, completely ad hoc approach, but resulting
          #outputs are much smoother.
          #perhaps akin to taking prior from z_score, hack conjugate prior?
          for index,column in enumerate(column_list):
               s_pdf_coeff,w_pdf_coeff=coeff_dict['sfit_'+column],coeff_dict['wfit_'+column]
               likelihood_s[:,index] = self.johnsonsu_pdf(norm_data[:,index],*s_pdf_coeff)
               likelihood_w[:,index] = self.johnsonsu_pdf(norm_data[:,index],*w_pdf_coeff)
               jsu_mean_s[:,index],jsu_var_s[:,index] = self.johnsonsu_stats(s_pdf_coeff)
               jsu_mean_w[:,index],jsu_var_w[:,index] = self.johnsonsu_stats(w_pdf_coeff)
               baseline_rand[:,index] = self.norm_pdf(norm_data[:,index])
               #store deviations from distribution means
               dev_s[:,index] = np.subtract(norm_data[:,index],jsu_mean_s[:,index])
               dev_w[:,index] = np.subtract(norm_data[:,index],jsu_mean_w[:,index])
               z_s = np.divide(dev_s[:,index],np.sqrt(jsu_var_s[:,index]))
               z_w = np.divide(dev_w[:,index],np.sqrt(jsu_var_w[:,index]))
               lz_s = self.norm_pdf(z_s)
               lz_w = self.norm_pdf(z_w)
               likelihood_s[:,index] = np.multiply(likelihood_s[:,index],lz_s)
               likelihood_w[:,index] = np.multiply(likelihood_w[:,index],lz_w)

          #linear ind likelihoods, sum logs for total
          np.clip(likelihood_s,0.0001,np.inf,out=likelihood_s)
          np.clip(likelihood_w,0.0001,np.inf,out=likelihood_w)
          np.clip(baseline_rand,0.0001,np.inf,out=baseline_rand)
          ll_s = np.nansum(np.log(likelihood_s),axis=1,dtype=np.float64)
          ll_w = np.nansum(np.log(likelihood_w),axis=1,dtype=np.float64)
          ll_rand = np.nansum(np.log(baseline_rand),axis=1,dtype=np.float64)
          #added to give log likelihood gain LLG
          llg_s = ll_s - ll_rand
          llg_w = ll_w - ll_rand
          llg_ratio = np.subtract(llg_s,llg_w)
          chisq_s = np.nansum(np.divide(np.multiply(dev_s,dev_s),jsu_var_s),axis=1)
          chisq_w = np.nansum(np.divide(np.multiply(dev_w,dev_w),jsu_var_w),axis=1)
          #store score as LLG ratio 

          data_array['cscore'] = llg_ratio.astype(np.float32)
          data_array['cllgS'] = llg_s.astype(np.float32)
          data_array['cllgW'] = llg_w.astype(np.float32)
          data_array['cchiS'] = chisq_s.astype(np.float32)
          data_array['cchiW'] = chisq_w.astype(np.float32)



     def score_breakdown(self,data_array,results_array):
          #boolean array of criteria
          #1 = label S
          #2 = label W
          #3 = obs S
          #4 = good S score
          #5 = good W score
          logical_ass = np.zeros((data_array.shape[0],5),dtype=np.bool_)
          # returns boolean masks array
          selectors = Selectors(data_array)
          logical_ass[:,0] = selectors.inc_obss_bool 
          logical_ass[:,1] = selectors.inc_obsw_bool 
          logical_ass[:,2] = results_array['score'] >= 0 
          logical_ass[:,3] = results_array['llgS'] > 0.0
          logical_ass[:,4] = results_array['llgW'] > 0.0
          #logical classes (probably not a logical approach . . . )
          lc10110 = (logical_ass == (1,0,1,1,0)).all(axis=1) #1 = TP w / bad water score
          lc10101 = (logical_ass == (1,0,1,0,1)).all(axis=1) #2 = impossible
          lc10111 = (logical_ass == (1,0,1,1,1)).all(axis=1) #3 = TP w / good water score
          lc10100 = (logical_ass == (1,0,1,0,0)).all(axis=1) #4 = TP with bad scores

          lc01010 = (logical_ass == (0,1,0,1,0)).all(axis=1) #5 = impossible
          lc01001 = (logical_ass == (0,1,0,0,1)).all(axis=1) #6 = TN with bad S score
          lc01011 = (logical_ass == (0,1,0,1,1)).all(axis=1) #7 = TN with good S score
          lc01000 = (logical_ass == (0,1,0,0,0)).all(axis=1) #8 = TN with bad scores

          lc01110 = (logical_ass == (0,1,1,1,0)).all(axis=1) #9 = FP with good S score (bad label?)
          lc01101 = (logical_ass == (0,1,1,0,1)).all(axis=1) #10 = impossible
          lc01111 = (logical_ass == (0,1,1,1,1)).all(axis=1) #11 = FP with good W score
          lc01100 = (logical_ass == (0,1,1,0,0)).all(axis=1) #12 = FP with bad scores

          lc10010 = (logical_ass == (1,0,0,1,0)).all(axis=1) #13 = impossible
          lc10001 = (logical_ass == (1,0,0,0,1)).all(axis=1) #14 = FN with good W score (bad label?)
          lc10011 = (logical_ass == (1,0,0,1,1)).all(axis=1) #15 = FN with good scores
          lc10000 = (logical_ass == (1,0,0,0,0)).all(axis=1) #16 = FN with bad scores
          #for peaks with neither s nor w label
          lc00010 = (logical_ass == (0,0,0,1,0)).all(axis=1) #17 other -- obsw,goods,badw --> impossible
          lc00001 = (logical_ass == (0,0,0,0,1)).all(axis=1) #18 other -- obsw,bads,goodw --> ok
          lc00011 = (logical_ass == (0,0,0,1,1)).all(axis=1) #19 other -- obsw,goods,betterw --> ok
          lc00000 = (logical_ass == (0,0,0,0,0)).all(axis=1) #20 other -- all bad --> ok

          lc00110 = (logical_ass == (0,0,1,1,0)).all(axis=1) #21 other obss,goods, badw --> ok
          lc00101 = (logical_ass == (0,0,1,0,1)).all(axis=1) #22 other obss, bads, goodw --> impossible
          lc00111 = (logical_ass == (0,0,1,1,1)).all(axis=1) #23 other obss, good, good --> ok
          lc00100 = (logical_ass == (0,0,1,0,0)).all(axis=1) #24 other obss, bad, bad --> ok

          result_class = np.zeros(data_array.shape[0],dtype=np.int16)
          for index,lclass in enumerate((lc10110,lc10101,lc10111,lc10100,
                                         lc01010,lc01001,lc01011,lc01000,
                                         lc01110,lc01101,lc01111,lc01100,
                                         lc10010,lc10001,lc10011,lc10000,
                                         lc00010,lc00001,lc00011,lc00000,
                                         lc00110,lc00101,lc00111,lc00100)):
               select = lclass
               result_class[select] = index + 1 #number from one
          results_array['rc'] = result_class


     def score_flags(self,data_array,full_out=False):
          #assigns a "result_class"
          #cutoffs for good values (g=good,s/w,density/contact,llg/chi)
          gses_cut = 0.0
          gscs_cut = -3.0
          gsec_cut = 30.0
          gscc_cut = 3.5
          gwes_cut = 0.0
          gwcs_cut = -2.5
          gwec_cut = 28.0
          gwcc_cut = 5.0
          
          #cutoffs for best by chiS, chiW, cchiS, cchiW
          #best_chi_w = [(10.0,45.0),(2.0,20.0),(3.0,10.0),(0.0,1.8)]
          #best_chi_s = [(2.0,22.0),(10.0,90.0),(0.0,2.0),(1.5,7.0)]
          #best_chi_o = [(5.0,27.0),(10.0,50.0),(0.0,3.0),(0.5,7.0)]
          #best_chi_m = [(5.0,45.0),(5.0,150.0),(7.0,17.0),(0.5,4.0)]

          goods_es = data_array['llgS'] > gses_cut
          goods_cs = data_array['cllgS'] > gscs_cut
          goods_ec = data_array['chiS'] < gsec_cut
          goods_cc = data_array['cchiS'] < gsec_cut
          goodw_es = data_array['llgW'] > gwes_cut
          goodw_cs = data_array['cllgW'] > gwcs_cut
          goodw_ec = data_array['chiW'] < gwec_cut
          goodw_cc = data_array['cchiW'] < gwec_cut

          preds = data_array['prob'] > 0.8 #prob not water
          predw = np.invert(preds)

          #logical assignments 
          #flags are prediction dependent (only S for pred S etc.)
          logical_ass = np.zeros((data_array.shape[0],5),dtype=np.int16)
          logical_ass[:,0] = preds
          logical_ass[:,1] = np.logical_or(preds.astype(np.int16)*goods_es,predw.astype(np.int16)*goodw_es)
          logical_ass[:,2] = np.logical_or(preds.astype(np.int16)*goods_cs,predw.astype(np.int16)*goodw_cs)
          logical_ass[:,3] = np.logical_or(preds.astype(np.int16)*goods_ec,predw.astype(np.int16)*goodw_ec)
          logical_ass[:,4] = np.logical_or(preds.astype(np.int16)*goods_cc,predw.astype(np.int16)*goodw_cc)


          if self.verbose:
               ppsel = Selectors(data_array)
               labs = ppsel.inc_obss_bool
               labw = ppsel.inc_obsw_bool
               nolabel = np.invert(np.logical_or(labs,labw))
               tpsel = np.logical_and(labs,preds)
               tnsel = np.logical_and(labw,predw)
               fpsel = np.logical_and(labw,preds)
               fnsel = np.logical_and(labs,predw)
               owsel = np.logical_and(nolabel,predw)
               ossel = np.logical_and(nolabel,preds)

               break_total = lambda sel: np.array([np.count_nonzero(np.logical_and(tpsel,sel)),
                                                   np.count_nonzero(np.logical_and(tnsel,sel)),
                                                   np.count_nonzero(np.logical_and(fpsel,sel)),
                                                   np.count_nonzero(np.logical_and(fnsel,sel)),
                                                   np.count_nonzero(np.logical_and(ossel,sel)),
                                                   np.count_nonzero(np.logical_and(owsel,sel))],dtype=np.int64)

               lc = 1
               print "GROUPS %3s %10s || %7s ||  %s" % ("GRP","PESCSECCC T","total","  S-S     W-W     W-S     S-W     0-S     0-W")
               for a in range(2):
                    for b in range(2):
                         for c in range(2):
                              for d in range(2):
                                   for e in range(2):
                                        flagsum = b+c+d+e
                                        intersection = (logical_ass == (a,b,c,d,e)).all(axis=1)
                                        count = np.count_nonzero(intersection)
                                        if count > -1:
                                             breakdown = break_total(intersection)

                                             print "COUNTS %3d %1d %1d %1d %1d %1d %1d || %7d ||  %s" % (lc,a,b,c,d,e,flagsum, count,
                                                       " ".join(('{:^7s}'.format(str(x)) for x in break_total(intersection))))

                                        lc = lc+1
          if full_out:
               return logical_ass

          flagsum = np.nansum(logical_ass[:,1::],axis=1)
          #zero by default, all junk peaks
          #pc00 = np.logical_or(flagsum < 2,logical_ass == (1,1,1,1,0)).all(axis=1) #junk peaks
          pc01 = (logical_ass == (1,1,1,1,1)).all(axis=1) #best SO4, all criteria met
          pc02 = (logical_ass == (1,0,1,1,1)).all(axis=1) #SO4, bad ed SO4 score
          pc03 = (logical_ass == (1,1,0,1,1)).all(axis=1) #S
          pc04 = (logical_ass == (1,1,1,0,1)).all(axis=1) #
          pc05 = (logical_ass == (0,0,0,1,1)).all(axis=1) #
          pc06 = (logical_ass == (0,1,1,0,1)).all(axis=1) #W, 
          pc07 = (logical_ass == (0,0,1,1,1)).all(axis=1) #W, but bad ED score?
          pc08 = (logical_ass == (0,1,0,1,1)).all(axis=1) #also water, bad water chi
          pc09 = (logical_ass == (0,1,1,1,1)).all(axis=1) #best water, all criteria met



          result_class = np.zeros(data_array.shape[0],dtype=np.int16)
          for index,lclass in enumerate((pc01,pc02,pc03,pc04,pc05,pc06,pc07,pc08,pc09)):
               select = lclass
               result_class[select] = index + 1 #number from one
          if self.verbose:
               total=0
               for i in range(np.amax(result_class)+1):
                    selector = result_class == i
                    count = np.count_nonzero(selector)
                    if count > 0:
                         print "                  RC %1d %10d || %s " % (i,count," ".join(('{:^7s}'.format(str(x)) for x in break_total(selector))))
                         total = total + count
               print "TOTAL",total,selector.shape[0]
               
          return result_class
          
     def score_class(self,numTP,numTN,numFP,numFN):
          count = float(numTP+numTN+numFP+numFN)
          if count == 0:
               return 0.0,0.0,0.0,0.0,0.0
          acc = (numTP + numTN)/count
          if (numTP + numFP) > 0:
               ppv = float(numTP)/(numTP + numFP)
          else:
               ppv = 0.0
          if (numTN + numFN) > 0:
               npv = float(numTN)/(numTN + numFN)
          else:
               npv = 0.0
          if (numTP + numFN) > 0:
               rec = float(numTP)/(numTP + numFN)
          else:
               rec = 0.0
          if (ppv + rec) > 0:
               #harmonic mean of ppv and rec
               f1 = 2.0*(ppv*rec)/(ppv+rec)
          else:
               f1 = 0.0
          return acc,ppv,npv,rec,f1

     def score_stats(self,data_array,plot=False):

          if plot:
               gridplot = plt.figure(figsize=(12,12))
               plot_data = []
          
          result_class = self.score_flags(data_array)

          for resbin in range(10):
               if resbin == 0: #all data
                    selected_data = data_array
                    selected_class = result_class
               else:
                    selected_data = data_array[data_array['bin'] == resbin]
                    selected_class = result_class[data_array['bin'] == resbin]
               count = selected_data.shape[0]
               if plot and count < 10: #not enough data in bin
                    plot_data.append([0,0,0,0,0,0,0,0,0,0])
                    break
               selectors = Selectors(selected_data)
               sall = np.ones(selected_data.shape[0],dtype=np.bool_)
               labs = selectors.inc_obss_bool
               labw = selectors.inc_obsw_bool
               labo = np.invert(np.logical_or(labs,labw))
               preds = selected_data['prob'] > 0.5
               predw = np.invert(preds)
               tpsel = np.logical_and(preds,labs)
               tnsel = np.logical_and(predw,labw)
               fpsel = np.logical_and(preds,labw)
               fnsel = np.logical_and(predw,labs)
               owsel = np.logical_and(predw,labo)
               ossel = np.logical_and(preds,labo)
               goods = selected_class == 1
               goodw = selected_class > 7
               goodf = selected_data['fc'] == 0
               badf = np.invert(goodf)

               print "STATS FOR BIN %1s PEAKS %8s" % (resbin,count)
               for name,selector in (('ALL ',sall),('GSO4',goods),
                                     ('GWAT',goodw),('GFLG',goodf),
                                     ('BFLG',badf)):
                    counts = []
                    bindat=selected_data[selector]
                    rcsel = selected_class[selector]
                    for rclass in (tpsel,tnsel,fpsel,fnsel,ossel,owsel):
                         counts.append(np.count_nonzero(np.logical_and(selector,rclass)))
                    acc,ppv,npv,rec,f1 = self.score_class(*counts[0:4])
                    total = np.clip(np.nansum(counts),1,np.inf)
                    fcnt= np.array(counts,dtype=np.float64)/total
                    print "    %s   TP %8s(%4.3f) TN %8s(%4.3f) FP %8s(%4.3f) FN %8s(%4.3f) OS %8s(%4.3f) OW %8s(%4.3f)" % (name,
                                                                                                                            counts[0],fcnt[0],
                                                                                                                            counts[1],fcnt[1],
                                                                                                                            counts[2],fcnt[2],
                                                                                                                            counts[3],fcnt[3],
                                                                                                                            counts[4],fcnt[4],
                                                                                                                            counts[5],fcnt[5])
                    print "         RATIOS: ACC %4.3f    PPV %4.3f    NPV %4.3f    REC %4.3f        F1 %4.3f" % (acc,ppv,npv,rec,f1)

                    if plot:
                         plot_data.append((acc,ppv,npv,rec,f1))
          if plot:
               plot_data = np.array(plot_data)
               for rbin in np.arange(10):
                    sub = gridplot.add_subplot(5,2,rbin+1)
                    bar_data1 = plot_data[2*rbin]
                    bar_data2 = plot_data[2*rbin+1]
                    sub.bar(np.arange(5)-0.2,bar_data1,width=0.2,align='center',color='r')
                    sub.bar(np.arange(5),bar_data2,width=0.2,align='center',color='b')
                    sub.set_xticks(np.arange(5))
                    sub.set_xticklabels(['ACC','PPV','NPV','REC','F1b'])
                    sub.set_ylabel("BIN_"+str(rbin))
               plt.savefig("SCORE_PLOT.png")
               plt.clf()
               plt.close()

                   
     def peak_summary(self,pdict):
          #checks for outliers (5sig) for scoring features
          resolution = pdict['resolution']
          col_list1 = list(col for col in self.ppsel(raw_data=None).std_view_col)
          col_list2 = ['c1','charge']
          coeff_dict = self.contact_coeffs
          rev_lookup = {}
          for pdk,npc in self.ppio.pdict_to_numpy.iteritems():
               rev_lookup[npc] = pdk
          outstr = []
          for index,column in enumerate(col_list1):
               resd_mean,resd_sig = self.get_res_scales(column,resolution)
               pd_data = pdict[rev_lookup[column]]
               pd_z = (pd_data - resd_mean)/np.exp(resd_sig)
               if np.abs(pd_z) >= 5.0:
                    outstr.append("PDATA OUTLIER: %12s %13s Column %12s Z= %5.2f" % (pdict['db_id'],pdict['resat'],column,pd_z))
          for index,column in enumerate(col_list2):
               pd_data = pdict[rev_lookup[column]]
               pd_z = np.divide(np.subtract(pd_data,coeff_dict['mean_'+column]),coeff_dict['std_'+column])
               if np.abs(pd_z) >= 5.0:
                    outstr.append("PDATA OUTLIER: %12s %13s Column %12s Z= %5.2f" % (pdict['db_id'],pdict['resat'],column,pd_z))
          if len(outstr) > 0:
               for str in outstr:
                    print str


     def peak_edc(self,results_array):
          #function to assign electron density class
          #1-8, 1=best s, 8=best water
          #cutoffs for "good" values for llg and chisq (empirical)
          #cutoffs for score/chi S/W
          gss_cut = -1.0 #good
          xss_cut = 1.0  #great
          gsc_cut = 35
          gws_cut = -0.5
          xws_cut = -1.5
          gwc_cut = 35
          s2f_cut = 4.0
          w2f_cut = 2.0
          #note, as of 07012017 draft, scores are res scaled, llg values are not

          preds = results_array['score'] > 0.0
          goods_ss = results_array['score'] > gss_cut
          greats_ss = results_array['score'] > xss_cut
          goods_sc = results_array['chiS'] < gsc_cut
          goods_den = results_array['2fofc_sigo'] > s2f_cut
          ps_gs = np.logical_and(preds,goods_ss)
          ps_xs = np.logical_and(preds,greats_ss)
          ps_gc = np.logical_and(preds,goods_sc)
          ps_gd = np.logical_and(preds,goods_den)

          predw = np.invert(preds)
          goodw_ws = results_array['score'] < gws_cut
          greatw_ws = results_array['score'] < xws_cut
          goodw_wc = results_array['chiW'] < gwc_cut
          goodw_den = results_array['2fofc_sigo'] > w2f_cut
          pw_gs = np.logical_and(predw,goodw_ws)
          pw_xs = np.logical_and(predw,greatw_ws)
          pw_gc = np.logical_and(predw,goodw_wc)
          pw_gd = np.logical_and(predw,goodw_den)
          
          good_score = np.logical_or(ps_gs,pw_gs)
          great_score = np.logical_or(ps_xs,pw_xs)
          good_chi = np.logical_or(ps_gc,pw_gc)
          good_den = np.logical_or(ps_gd,pw_gd)

          scr_table = np.zeros((results_array.shape[0],4),dtype=np.int8)
          scr_table[:,0] = good_den
          scr_table[:,1] = good_score
          scr_table[:,2] = good_chi
          scr_table[:,3] = great_score
          
          logical_ass = np.zeros((results_array.shape[0],2),dtype=np.int16)
          logical_ass[:,0] = preds
          logical_ass[:,1] = np.nansum(scr_table,axis=1)

          #explicit def of all edc
          edc = np.zeros(results_array.shape[0],dtype = np.int16)
          
          edc1 =(logical_ass == (1,4)).all(axis=1) #all 4 criteria good great S
          edc2 =(logical_ass == (1,3)).all(axis=1) #3 out of 4 --> good S
          edc3 =(logical_ass == (1,2)).all(axis=1) #2 out of 4 --> weak S
          edc4 =(logical_ass == (1,1)).all(axis=1) #1 out of 4 --> bad S
          edc5 =(logical_ass == (0,1)).all(axis=1) #etc
          edc6 =(logical_ass == (0,2)).all(axis=1) #
          edc7 =(logical_ass == (0,3)).all(axis=1) #
          edc8 =(logical_ass == (0,4)).all(axis=1) #
          #will remain zero for "junk" peaks


          for edc_ind,selector in enumerate((edc1,edc2,edc3,edc4,
                                             edc5,edc6,edc7,edc8)):
               edc[selector] = edc_ind + 1

          return edc

     def peak_fc(self,data_array):
          #function to assign a class for various peak flags
          # 0 = no flags
          # 1 = special position
          # 2 = very bad contacts
          # 3 = bad contacts and one close contact
          # 4 = bad contacts
          # 5 = one close contact
          # 6 = weak 2fofc (less than scaled 0.5sigma)
          # 7 = remote, far from any contact

          flags_col=['weak','remote','close','special','badc','sadc']
          flags_fmt = [np.bool_,np.bool_,np.bool_,np.bool_,np.bool_,np.bool_]
          flags_dtype = np.dtype(zip(flags_col,flags_fmt))
          flags_arr = np.zeros(data_array.shape[0],dtype=flags_dtype)

          #flag weak peaks with low 2fofc level (scaled)
          solvent_content = np.clip(data_array['solc'],0.2,0.8)
          sig_scale = 0.5*np.sqrt(0.5/(1.0 - solvent_content))
          flags_arr['weak'] = data_array['2fofc_sigo'] < sig_scale

          csum = np.zeros(data_array.shape[0],dtype=np.int16)
          for ccol in ('ol','om','oh','sl','sm'):
               csum = csum+data_array[ccol]
          notfar_c1 = (data_array['c1'] < 4.5).astype(np.int16)
          csum=csum+notfar_c1
          flags_arr['remote'] = csum == 0 #no contacts to anything
          flags_arr['special'] = data_array['sp'] > 0 #likely special position

          #screen for bad contacts
          vbarr = np.zeros(data_array.shape[0],dtype=np.bool_) #very bad
          mbarr = np.zeros(data_array.shape[0],dtype=np.bool_) #moderate bad
          s1 = data_array['wl'] >= 2
          s2 = np.logical_and(data_array['wl'] == 1,data_array['wm'] > 1)
          s3 = np.logical_and(data_array['wl'] == 1,data_array['st'] > 8)
          s4 = data_array['wm'] >= 3
          vbcut = np.logical_or(s1,np.logical_or(s2,np.logical_or(s3,s4)))
          vbarr[vbcut] = True
          mbarr[vbcut] = True

          s1 = data_array['wl'] >= 1
          s2 = np.logical_and(data_array['wl'] == 1,data_array['wm'] >= 1)
          s3 = data_array['st'] > 4
          s4 = data_array['wt'] > 2
          mbcut = np.logical_or(s1,np.logical_or(s2,np.logical_or(s3,s4)))
          mbarr[mbcut] = True
          
          flags_arr['badc'] = vbcut
          flags_arr['sadc'] = mbcut
          flags_arr['close'] = data_array['c1'] < 2.2


          #assign flags
          flag_class = np.zeros(flags_arr.shape[0],dtype=np.int16)
          sp_sel = flags_arr['special'] == True
          badc_sel = flags_arr['badc'] == True
          sadc_sel = flags_arr['sadc'] == True
          close_sel = flags_arr['close'] == True
          remote_sel = flags_arr['remote'] == True
          weak_sel = flags_arr['weak'] == True

          #assign in rev order of precidence (remote < special, special implies badc, etc.)
          flag_class[remote_sel] = 7
          flag_class[weak_sel] = 6
          flag_class[close_sel] = 5
          flag_class[sadc_sel] = 4
          flag_class[np.logical_and(sadc_sel,close_sel)] = 3
          flag_class[badc_sel] = 2
          flag_class[sp_sel] = 1

          return flag_class


     def peak_cc(self,data_array):
          #function to give a class for local contact environment


          #cutoffs for "good" values for llg and chisq (empirical)
          #cutoffs for score/chi S/W
          gss_cut = -3.0
          xss_cut = -1.0
          gsc_cut = 3.0
          gws_cut = -4.0
          xws_cut = -2.0
          gwc_cut = 5.0
    
          preds = data_array['prob'] > 0.8

          goods_ss = data_array['cllgS'] > gss_cut
          greats_ss = data_array['cllgS'] > xss_cut
          goods_sc = data_array['cchiS'] < gsc_cut
          ps_gs = np.logical_and(preds,goods_ss)
          ps_xs = np.logical_and(preds,greats_ss)
          ps_gc = np.logical_and(preds,goods_sc)
          
          predw = np.invert(preds)
          goodw_ws = data_array['cllgW'] > gws_cut
          greatw_ws = data_array['cllgW'] > xws_cut
          goodw_wc = data_array['cchiW'] < gwc_cut
          pw_gs = np.logical_and(predw,goodw_ws)
          pw_xs = np.logical_and(predw,greatw_ws)
          pw_gc = np.logical_and(predw,goodw_wc)

          good_score = np.logical_or(ps_gs,pw_gs)
          great_score = np.logical_or(ps_xs,pw_xs)
          good_chi = np.logical_or(ps_gc,pw_gc)

          one_good_score = np.logical_xor(good_chi,good_score)
          two_good_scores = np.logical_and(good_chi,good_score)
          

          logical_ass = np.zeros((data_array.shape[0],4),dtype=np.bool_)
          logical_ass[:,0] = preds
          logical_ass[:,1] = one_good_score
          logical_ass[:,2] = two_good_scores
          logical_ass[:,3] = np.logical_and(great_score,two_good_scores)

          c_class = np.zeros(data_array.shape[0],dtype = np.int16)
          
          cc1 =  (logical_ass == (1,0,1,1)).all(axis=1) #preds, good chi, great score
          cc2 =  (logical_ass == (1,0,1,0)).all(axis=1) #preds, 2 good scores
          cc3 =  (logical_ass == (1,1,0,0)).all(axis=1) #preds, one good score
          cc4 =  (logical_ass == (1,0,0,0)).all(axis=1) #preds, no good scores (junk)
          cc5 =  (logical_ass == (0,0,0,0)).all(axis=1) #predw, no good scores (junk)
          cc6 =  (logical_ass == (0,1,0,0)).all(axis=1) #predw, one good score
          cc7 =  (logical_ass == (0,0,1,0)).all(axis=1) #predw, two good scores
          cc8 =  (logical_ass == (0,0,1,1)).all(axis=1) #predw, good chi, great scores

          for cc_ind,selector in enumerate((cc1,cc2,cc3,cc4,
                                            cc5,cc6,cc7,cc8)):
               c_class[selector] = cc_ind + 1
          return c_class


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


     def bad_contacts(self,all_peak_db):
          for unal,pdict in all_peak_db.iteritems():
               if pdict['model'] in [1,2] or pdict['status'] in [1,3,6,7]:
                    continue
               ori_status = pdict['status']
               fc = pdict['fc']
               edc = pdict['edc']
               pnw = pdict['prob']
               pick = pdict['pick']
               c1 = pdict['c1']
               cscore = pdict['cscore']
               shortest_worst = pdict['worst_mm']
               if fc in [0,1]: #no problems
                    continue
               if fc == 2 or fc == 3: #bad clashes
                    #most model errors have ed score > 0:
                    if pdict['score'] > 0  or pdict['cscore'] > 0:
                         if shortest_worst['name'] not in ['N','C','CA','O']:
                              pdict['status'] = 401
                         else:
                              pdict['status'] = 402
                    else:
                         pdict['status'] = 403

               if fc == 4: #less bad contacts
                    if pnw > 0.8: 
                         if pdict['edc'] < 7 and pdict['fofc_sig_out'] > 2.0:
                              pdict['status'] = 8001
                         else:
                              pdict['status'] = 410
                    elif pdict['score'] > 0:
                         if shortest_worst['name'] not in ['N','C','CA','O']:
                              pdict['status'] = 411
                         else:
                              pdict['status'] = 412
                    else:
                         pdict['status'] = 413

               if fc == 5: #one close contact
                    if pnw > 0.8: 
                         if pdict['edc'] < 7:
                              pdict['status'] = 421
                         else:
                              pdict['status'] = 422
                    elif pnw > 0.5:
                         if pdict['edc'] > 6:
                              pdict['status'] = 423
                         else:
                              pdict['status'] = 424
                    else:
                         if pick != 1: #allow close water to pass thru
                              if c1 < 1.8:
                                   pdict['status'] = 422
                              else:
                                   if pnw < 0.10:
                                        pdict['status'] = 1001
                                        pdict['pick'] = 1
                                        pdict['tflag'] = 1 #fix pick
                                   else:
                                        pdict['status'] = 422
               if fc == 6: #weak
                    pdict['status'] = 431
               if fc == 7: #remote
                    anchor = pdict.get('anchor',None)
                    if anchor is not None:
                         if anchor['model'] == 3:
                              pdict['status'] = 441
                    else:
                         pdict['status'] = 442
               reassign = False
               #metals can be incorrectly flagged, re-examine
               if pdict['status'] in [401,421]:
                    if c1 < 1.80:
                         reassign = False
                    else:
                         if pnw < 0.80:
                              reassign = False
                         else:
                              reassign = True
               if pdict['status'] == 8001:
                    if c1 < 2.4:
                         reassign = True
               if reassign:
                    pdict['status'] = ori_status

                    


     def update_all_models(self,all_peak_db):
          peak_list = list(pdict['unal'] for pdict in all_peak_db.values() if (pdict['model'] == 4 and pdict['status'] != 7))
          proc_sol = list(pdict['unal'] for pdict in all_peak_db.values() if (pdict['model'] == 3 and pdict['status'] not in [1,3,6,7]))
          nproc_sol = list(pdict['unal'] for pdict in all_peak_db.values() if (pdict['model'] == 3 and pdict['status'] == 6))
          for uind,unal in enumerate(peak_list):
               pdict = all_peak_db[unal]
               self.update_point_model(pdict)
               #u2r = lambda x: all_peak_db.get(x,{'resat':'Null'})['resat']
               #print "MOD",pdict['mflag'],pdict['resat']," ".join(list("%s %s" % (u2r(u),d) for u,d in pdict['sol_mod']))
               if pdict['mflag'] == 1 or len(pdict['sol_mod']) == 0:
                    continue
               peak_is_clust = pdict['mf'] % 10 > 2
               s1_pdict = all_peak_db[pdict['sol_mod'][0][0]]
               s1_clust = s1_pdict.get('clust_mem',[])
               mod_is_ma = len(s1_clust) > 0
               if not peak_is_clust and not mod_is_ma:
                    continue
               ma_to_peak = []
               if mod_is_ma:
                    for mod_u in s1_clust:
                         m_pdict = all_peak_db.get(mod_u,{'peak_contacts':[]})
                         pcont = m_pdict['peak_contacts']
                         ma_to_peak.extend(list(cont['unal'] for cont in pcont))
                    ma_to_peak = list(set(ma_to_peak))
                    pdict['modexp_clust'] = ma_to_peak
                    pdict['mflag'] = pdict['mflag'] + 100
               peak_to_sol = []
               if peak_is_clust:
                    cmem = pdict['clust_mem']
                    for m_unal in cmem:
                         s_pdict = all_peak_db[m_unal]
                         scont = s_pdict['sol_contacts']
                         peak_to_sol.extend(list(cont['unal'] for cont in scont))
                    peak_to_sol = list(set(peak_to_sol))
                    pdict['modexp_sol'] = peak_to_sol
                    pdict['mflag'] = pdict['mflag'] + 200

               #u2r = lambda x: all_peak_db.get(x,{'resat':'Null'})['resat']
               #print "CHECK",pdict['resat']," ".join(list(u2r(x) for x in ma_to_peak)),"||"," ".join(list(u2r(x) for x in peak_to_sol))
     """
     def update_mol_model(self,pdict):
          all_peak_db=pdict['peak_unal_db']
          ma_to_peak = []
          if mod_is_ma:
               for mod_u in s1_clust:
                         m_pdict = all_peak_db.get(mod_u,{'peak_contacts':[]})
                         pcont = m_pdict['peak_contacts']
                         ma_to_peak.extend(list(cont['unal'] for cont in pcont))
                    ma_to_peak = list(set(ma_to_peak))
               peak_to_sol = []
               if peak_is_clust:
                    cmem = pdict['clust_mem']
                    for m_unal in cmem:
                         s_pdict = all_peak_db[m_unal]
                         scont = s_pdict['sol_contacts']
                         peak_to_sol.extend(list(cont['unal'] for cont in scont))
                    peak_to_sol = list(set(peak_to_sol))

          
     """
     def update_point_model(self,pdict):
          #assigns mflag based on single closest possible model
          unal = pdict['unal']
          all_peak_db = pdict['peak_unal_db']
          if len(pdict['sol_mod']) == 0:
               return
          sol_mod = pdict['sol_mod']
          # collect booleans for assignment
          sm_unal = list(ud[0] for ud in sol_mod)
          recip_mf_list = list(all_peak_db.get(s_unal,{'mod_for':[(None,-99.99),]})['mod_for'] for s_unal in sm_unal)
          recip = list(mfl[0][0] == unal for mfl in recip_mf_list)
          rlen = list(len(mfl) for mfl in recip_mf_list)
          fdpos = list(ud[1] > 0 for ud in sol_mod)
          vshort = list(mfl[0][1] < 0.7 for mfl in recip_mf_list)
          short = list(mfl[0][1] < 1.7 for mfl in recip_mf_list)
          truth_table = np.zeros((4,len(sol_mod)),dtype=np.int)
          for tind,tdat in enumerate((recip,fdpos,short,vshort)):
               truth_table[tind] = tdat

          if len(sol_mod) == 1:
               if (truth_table == True).all():
                    pdict['mflag'] = 2 #one recip tight match
                    return
               if (truth_table[0:3] == True).all():
                    pdict['mflag'] = 3 #one recip dist match
                    return 
               if recip[0] and fdpos[0]: #just too far away
                    pdict['mflag'] == 1
                    return
               if not recip[0] and not fdpos[0]:
                    pdict['mflag'] = 4 #not recip or already claimed
                    return
               if fdpos[0]:
                    pdict['mflag'] = 8
                    
               return #unknown, remains zero
          if len(sol_mod) > 1:
               #look at first two sol_mod
               dlist = list(mfl[0][1] for mfl in recip_mf_list)
               if (truth_table[0:3,0:2] == True).all(): #two recip matches, both short and ass
                    if abs(dlist[1]-dlist[0]) < 0.5:
                         pdict['mflag'] = 5 #sym split
                         return
                    else:
                         pdict['mflag'] = 6 #asym split
                         return
               if (truth_table[0:2,0:2] == True).all(): #not short
                    n_short = np.count_nonzero(truth_table[2,0:2])
                    n_vshort = np.count_nonzero(truth_table[3:0:2])
                    if n_vshort == 1 and n_short == 1:
                         pdict['mflag'] = 2 #discard far peak
                         return
                    elif n_short == 1:
                         pdict['mflag'] = 6
                         return
               if recip[0] or recip[1]:
                    if short[0] and short[1]:
                         pdict['mflag'] = 7
                         return

          """
          print "RECIP",pdict['resat'],recip,fdpos,rdpos,vshort,short,rlen
          recip1_pdict = all_peak_db.get(sol_mod[0][0],None)
          if recip1_pdict is not None:
               
               mod_for = list(ud for ud in recip1_pdict['mod_for'] if ud[1] > 0)
               if mod_for[0][0] == unal: #recip match
                    if len(sol_mod) == 1 and len(mod_for) == 1:
                         mod_u, mod_d = mod_for[0]
                         #close, unambiguous model
                         if mod_d > 0.0 and mod_d < 0.7:
                              pdict['mflag'] = 2
                              return
                         #bit far, 
                         elif mod_d > 0.0 and mod_d < 1.7:
                              pdict['mflag'] = 3
                              return
                         elif mod_d < 0.0:
                              pdict['mflag'] = 4 #different peak
                              return
                    elif len(sol_mod) > 1: #multiple models for single peak 

                         mod_u1, mod_d1 = sol_mod[0]
                         mod_u2, mod_d2 = sol_mod[1]
                         m1_mf1 = all_peak_db[mod_u1]['mod_for'][0] #closest recip for 1st
                         m2_mf1 = all_peak_db[mod_u2]['mod_for'][0] #closest recip for 2nd
                         if m2_mf1[0] == unal: #2nd also recip match
                              ddiff = abs(mod_d1 - mod_d2)
                              if ddiff < 0.5:
                                   pdict['mflag'] = 5 #clear split
                              else:
                                   pdict['mflag'] = 6 #distant split
                         else:
                              pdict['mflag'] = 7 #2nd not matching
                    elif len(mod_for) > 1 and len(sol_mod) == 1:
                         peak_u1, peak_d1 = mod_for[0]
                         peak_u2, peak_d2 = mod_for[1]
                         p2_sm1 = all_peak_db[peak_u2]['sol_mod'][0] #closest recip for 2nd
                         if p2_sm1[0] == sol_mod[0]:
                              pdict['mflag'] = 8
                         else:
                              pdict['mflag'] = 9
          
               else:
                    pdict['mflag'] = 10
          """

     def update_anchors(self,all_peak_db,allow_cross=False):
          # 1) scores peaks and updates status
          # 2) rescores all peaks with a c1 value of 2.2
          # 3) if 'prob' does drops below 0.2, likely can be water
          # 4) selects peaks marked as not water but could be water
          #    and searchers for a closer anchor to update c1
          self.score_peaks(all_peak_db)
          self.ws_pass1(all_peak_db)
          anc_updates = 0
          mm_ulist = self.class_mismatches(all_peak_db)
   

          for unal in mm_ulist:
               pdict = all_peak_db[unal]
               update = False
               ori_c1 = pdict.get('c1',99.9)
               tmp = float(ori_c1)
               allowed_models = [pdict['model'],]
               if allow_cross and pdict['model'] == 4:
                    allowed_models.append(3)
               elif allow_cross and pdict['model'] == 3:
                    allowed_models.append(4)
               if pdict['mf'] % 10 == 0:
                    local_clust = []
               else:
                    local_clust = pdict.get('clust_mem',[])
               culled_anc = []
               if len(pdict['anc_cont']) > 0:
                    for anc_cont in sorted(pdict['anc_cont'], key = lambda x: x['distance'],reverse=True):
                         anc_pdict = all_peak_db.get(anc_cont['unal'],None)
                         if anc_pdict is None:
                              continue
                         #criteria for updating anchor,valid peak, not in self cluster
                         allowed = anc_pdict['model'] in allowed_models 
                         valid_s = anc_pdict['status'] > 1099 and anc_pdict['status'] < 5000
                         valid_f = anc_pdict.get('fc',0) not in [2,3,6,7]
                         shorter = anc_cont['distance'] < (ori_c1 - 0.1)
                         anchored = anc_pdict['c1'] < ori_c1
                         if anc_pdict['unal'] in local_clust:
                              noclust = False
                              if all((allowed,valid_s,valid_f,shorter,anchored)):
                                   print "MODANC_CHECK",pdict['resat'],pdict['status'],anc_pdict['resat'],anc_pdict['status'],anc_cont['distance']
                         else:
                              noclust = True
                         if all((allowed,valid_s,valid_f,shorter,noclust,anchored)):
                              culled_anc.append(anc_cont)
               if len(culled_anc) > 0:
                    new_anchor = culled_anc[0]
                    nanc_pdict = all_peak_db[new_anchor['unal']]
                    anc_for = nanc_pdict.get('anc_for',[])
                    anc_for.append(pdict['unal'])
                    nanc_pdict['anc_for'] = list(set(anc_for))
                    update = True
                    pdict['c1'] = new_anchor['distance']
                    ori_c1 = new_anchor['distance']
                    pdict['anchor'] = new_anchor
               if update:
                    anc_updates = anc_updates + 1
                    print "UPANC",pdict['resat'],pdict['status'],ori_c1,tmp,pdict['anchor']['resat']
          return anc_updates


     def ws_pass1(self,all_peak_db):
          #assigns status based on quality of pick, wsom
          #models and anchors initially ignored
          ppcont = self.ppcont

          for unal,pdict in all_peak_db.iteritems():
               if pdict['model'] in [1,2] or pdict['status'] in [-1,1,2,3,6,7]:
                    continue
               if pdict['status'] > 399 and pdict['status'] < 499:
                    continue #ignore flagged peaks that are bad
               filter_mask = pdict['filter_mask']
               filter_cuts = (6,6,6,3)
               edc = pdict['edc']
               cc = pdict['cc']
               rc = pdict['rc']
               fc = pdict['fc']
               pick = pdict['pick']
               ambig = pdict['ambig']
               probs = pdict['prob_data']
               pnw = pdict['prob']
               pnw_match = (pnw > 0.85 and pick > 1) or (pnw < 0.15 and pick == 1)
               not_filt = filter_mask[pick-1] < filter_cuts[pick-1]
               good_w = np.nansum(probs[0,0]) > 0.0 and pick == 1 and not_filt
               good_s = np.nansum(probs[0,1]) > 0.0 and pick == 2 and not_filt
               good_o = np.nansum(probs[0,2]) > 0.0 and pick == 3 and not_filt
               good_m = np.nansum(probs[0,3]) > 0.0 and pick == 4 and not_filt
               any_good = good_w or good_s or good_o or good_m
               new_status = 1000*pick+400*pnw_match+200*(not ambig)+100*any_good
               
               if pdict['status'] != new_status:
                    pdict['status'] = new_status 

     def class_mismatches(self,all_peak_db):
          #if c1 is not set correctly (no good anchor found)
          #classification can be off, or it may not matter
          #try a c1 distance of 2.2 
          temp_db = {}
          mismatch_ulist = []
          for unal,pdict in all_peak_db.iteritems():
               if pdict['status'] not in  [-1,1,2,3,6,7] and pdict['c1'] > 2.2:
                    #check all peaks, but only labeled water
                    if pdict['model'] == 4 or ( pdict['model'] == 3 and pdict['label'] == 1):
                         temp_db[unal] = pdict.copy()
                         temp_db[unal]['c1'] = 2.2
          #if picks change upon shortening of c1, mark as mismatch
          new_plist = list(pdict for pdict in temp_db.values())
          scan_array = self.ppio.extract_raw(new_plist)
          #copied from tasks
          self.contact_da(scan_array)
          self.standardize_data(scan_array,post_pca=True,composite=True)
          self.chi_prob(scan_array)
          self.ppio.update_from_np(temp_db,scan_array)
          self.score_peaks(temp_db)
          for unal in temp_db.keys():
               pd1 = all_peak_db[unal]
               pd2 = temp_db[unal]
               pick1 = pd1['pick']
               prob2 = pd2['prob']
               if pick1 > 1 and prob2 < 0.2:
                    mismatch_ulist.append(unal)
          #if edc and cc mismatch, also mark
          #if c1 distance out of range for water pick
          for unal,pdict in all_peak_db.iteritems():
               if pdict['status'] not in  [-1,1,2,3,6,7] and pdict['c1'] > 2.2:
                    if pdict['model'] == 4 or (pdict['model'] == 3 and pdict['label'] == 1):
                         if pdict['edc'] in [6,7,8] and pdict['cc'] in [1,2,3]:
                              mismatch_ulist.append(unal)
                         if pdict['c1'] > 3.5 and pdict['pick'] == 1:
                              mismatch_ulist.append(unal)
          mismatch_ulist = list(set(mismatch_ulist))
          print "MISMATCHES"," ".join(list(all_peak_db[unal]['resat'] for unal in mismatch_ulist))
          return mismatch_ulist

     def score_peaks(self,db_of_peaks):
          for pdict in db_of_peaks.values():
               if pdict['status'] not in [-1,1,2,3,6,7]:
                    probs = pdict['prob_data']
                    if 'resid_names' not in pdict:
                         resid_names = pdict['master_dict']['kde']['populations']
                    else:
                         resid_names = pdict['resid_names']
                    if pdict['tflag'] == 1: #fixed pick
                         pick1 = pdict['pick']
                    else:
                         pick1 = self.pput.pick_from_prob(probs)
                         #hack to cull mispicked metals,force another pick
                         if pick1 == 4 and (pdict['c1'] > 3.1 or pdict['cscore'] > 1.0):
                              nprobs = probs.copy()
                              nprobs[:,3] = -99.9
                              pick1 = self.pput.pick_from_prob(nprobs)
                    best_picks = np.argmax(probs,axis=1)+1
                    if (best_picks == pick1).all():
                         ambig = False
                    else:
                         ambig = True
                    pick_name = resid_names[pick1-1]
                    pdict['pick'] = pick1
                    pdict['pick_name'] = pick_name
                    pdict['ambig'] = ambig

     def chi_prob(self,data_array):
          #Logistic Model on difference between sum Chisq for SO4 and HOH
          #gives a "probability of "not water"
          if self.chiD_coeffs is None:
               sys.exit("NO chiD COEFFICIENTS, ERROR!")
          chiS = np.add(data_array['chiS'],data_array['cchiS'])
          chiW = np.add(data_array['chiW'],data_array['cchiW'])
          chiD = np.subtract(chiS,chiW)
          coef = self.chiD_coeffs
          prob = 1.0/(1.0+np.exp(-(coef[0]+chiD*coef[1])))
          data_array['prob'] = prob
