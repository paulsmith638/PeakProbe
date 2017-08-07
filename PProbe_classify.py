import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
from PProbe_selectors import Selectors
from PProbe_stats import StatFunc
from PProbe_matrix import PCA
#for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

class ClassifierFunctions:
     """
     uses the some of the same functions as the FitFunctions class in PPutil, but not the fitting functions
     or anything that requires scipy, whose import conflicts with things in phenix.python
     """
     def __init__(self,verbose=False):
          self.verbose = verbose
          self.ppsel = Selectors
          self.ppstat = StatFunc()
          self.pppca = PCA()
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
          self.jsu_coeffs = None

     def get_stats(self,array):
          if array.shape[0] == 0:
               mean,stdev = 0.0,1.0
          else:
               mean,stdev = np.nanmean(array),np.nanstd(array)
          if stdev == 0: #avoid divide by zero for only one point?  
               stdev = 1.0
          return float(mean),float(stdev)

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

     def get_jsu_coeffs(self,column_num,resolution):
          if self.jsu_coeffs is None:
               self.read_jsu_coeff()
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
               try:
                    print "READING PCA COEFF FROM FILE"
                    pca_matrix_coeff_file = open("pprobe_pca_matrix_coeff.dict",'r')
                    self.resscale_pca_coeffs = ast.literal_eval(pca_matrix_coeff_file.read())
                    pca_matrix_coeff_file.close()
               except:
                    sys.exit("CANNOT OPEN FILE: pprobe_pca_matrix_coeff.dict -- TERMINATING!")
          pca_coeffs = self.resscale_pca_coeffs
          num_col = int(np.sqrt(len(pca_coeffs)))
          #initialze a matrix
          xform_matrix = np.zeros((num_col,num_col))
          for i in range(xform_matrix.shape[0]):
               for j in range(xform_matrix.shape[1]):
                    column_index=str(i)+"_"+str(j)
                    params = pca_coeffs[column_index]
                    xform_matrix[i,j] = self.spline4k(params,res.reshape(-1))
          #these regenerated modal matrices are often have det != 1.0 and introduce
          #strange artifacts into the data
          #the following trick might work:
          modal_u,modal_l,modal_v = np.linalg.svd(xform_matrix)
          modal_fixed = np.dot(modal_u,modal_v)
          #possibility of reflections?
          orig_det = np.linalg.det(xform_matrix)
          fixed_det = np.linalg.det(modal_fixed)
          eval_sum = np.nansum(modal_l) #should be 3, tr(M), but isn't sometimes
          print "     PCA XFORM MATRIX RES %.2f DETin %.4f DETout %.4f TRACE %.4f" % (res,orig_det,fixed_det,eval_sum)
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



     def standardize_data(self,data_array,post_pca=False):
          """
          centers data by subtracting the resolution dependent s/w mean and sigma
          centering puts zero as the midpoint between sulfate and water means
          """
          print "STANDARDIZING DATA:"
          selectors=Selectors(data_array)
          #read in the dictionary of spline coefficients for raw data scaling


          #remove peaks not flagged as omit
          data_array = data_array[selectors.included_data_bool]
          scaled_array=data_array.copy()
          resolution_column = data_array['res']
          #method works with data both before and after PCA transformation
          #setup accordingly with array column names and scale coeff dictionaries
          if post_pca:
               column_list = selectors.pca_view_col
               if self.scale_post_pca_spline_coeffs is None:
                    try:
                         scale_post_pca_file = open("pprobe_post_pca_resscales.dict",'r')
                         self.scale_post_pca_spline_coeffs = ast.literal_eval(scale_post_pca_file.read())
                         scale_post_pca_file.close()
                    except:
                         sys.exit("CANNOT OPEN FILE: pprobe_post_pca_resscales.dict -- TERMINATING!")

          else:
               column_list = selectors.std_view_col
               if self.scale_raw_feature_spline_coeffs is None:
                    try:
                         scale_raw_feature_file = open("pprobe_pre_pca_resscales.dict",'r')
                         self.scale_raw_feature_spline_coeffs = ast.literal_eval(scale_raw_feature_file.read())
                         scale_raw_feature_file.close()

                    except:
                         sys.exit("CANNOT OPEN FILE: pprobe_pre_pca_resscales.dict -- TERMINATING!")

          for column in column_list:
               calc_meanx,calc_stdev = self.get_stats(data_array[column])
               if self.verbose:
                    print "      INPUT COLUMN %7s MEAN %4.2f SIG %4.2f" % (column,calc_meanx,calc_stdev)
               if post_pca == False:
                    resd_mean,resd_sig = self.get_res_scales(column,resolution_column)
               else:
                    resd_mean,resd_sig = self.get_post_pca_scales(column,resolution_column)

               scaled_array[column] = np.divide(np.subtract(scaled_array[column],resd_mean),np.exp(resd_sig))

               #check normalization applied correctly
               calc_meanx,calc_stdev = self.get_stats(scaled_array[column])
               if self.verbose:
                    print "     OUTPUT COLUMN %7s MEAN %4.2f SIG %4.2f" % (column,calc_meanx,calc_stdev)
          print "SCALED %s ROWS DATA" % scaled_array.shape[0]
          return scaled_array


     def pca_xform_data(self,norm_data,plot=False):
          """
          carries out PCA transformation, first generates matrix for PCA transformation given
          an input resolution, then applies it to one or more data points
          data must be sorted by resolution
          """
          
          print "PCA TRANSFORMATION:"
          norm_data = np.sort(norm_data,order=['res','id'])
          selectors=Selectors(norm_data)
          num_data = norm_data[selectors.std_view_col].view(selectors.raw_dtype)
          res = norm_data['res']
          xformed_data = np.zeros(num_data.shape[0],dtype=selectors.pca_view_dtype)
          #read in PCA transformation matrix coefficients
          #each entry is an i,j matrix entry with 6 spline coefficients used to calculate the eigenvector component at input resolution          
          if self.resscale_pca_coeffs is None:
               try:
                    pca_matrix_coeff_file = open("pprobe_pca_matrix_coeff.dict",'r')
                    self.resscale_pca_coeffs = ast.literal_eval(pca_matrix_coeff_file.read())
                    pca_matrix_coeff_file.close()
               except:
                    sys.exit("CANNOT OPEN FILE: pprobe_pca_matrix_coeff.dict -- TERMINATING!")
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
          array_size = num_data.shape[0]
          print "TRANSFORMED %s ROWS DATA" % array_size
          raw_array=xformed_data[selectors.pca_view_col].view(selectors.raw_dtype).copy()
          xformed_total_array = np.zeros(num_data.shape[0],dtype=selectors.alldata_pca_dtype)
          #recombine info data and transformed data into final array
          #numpy complains too much about writing to "views" of multiple fields, so . . . 
          for colname in selectors.alldata_pca_col:
               if colname in xformed_data.dtype.names:
                    xformed_total_array[colname] = xformed_data[colname]
               if colname in norm_data.dtype.names:
                    xformed_total_array[colname] = norm_data[colname]
          return xformed_total_array


     def read_jsu_coeff(self):
          #read in JSU coefficient dictionary
          if self.jsu_coeffs is None:
               try:
                    jsu_coeff_file = open("pprobe_jsu_coeffs.dict",'r')
                    self.jsu_coeffs = ast.literal_eval(jsu_coeff_file.read())
                    jsu_coeff_file.close() 
               except:
                    sys.exit("CANNOT OPEN FILE: pprobe_jsu_coeffs.dict -- TERMINATING!")




     #classifying an entire large dataset requires huge memory, break down by batch numbers (0-999) into 50 batches
     def batch_da(self,data_array,results_array,plot=False):
          for modbatch in np.arange(0,50,1):
               sel_mask = data_array['batch'] % 50 == modbatch
               input_data = data_array[sel_mask]
               input_results = self.initialize_results(input_data)
               sel_ind = np.argwhere(sel_mask == True)
               self.discriminant_analysis(input_data,input_results,plot=plot)
               results_array[sel_ind] = input_results #recombine output to original array
          return results_array

     def initialize_results(self,data_array):
          rows = data_array.shape[0]
          dtype = [('id','S16'),('res','f4'),('score','f4'),('prob','f4'),
                   ('llgS','f4'),('llgW','f4'),('chiS','f4'),('chiW','f4'),     
                   ('fchi','f4'),('kchi','f4'),('rc','i1')]
          return np.zeros(rows,dtype=dtype)

     def discriminant_analysis(self,data_array,results_array,plot=False):
          print "     SCORING %s PEAKS" % data_array.shape[0]
          selectors = Selectors(data_array)
          #clip resolution to range with proper training, minimal extrapolation
          resolution = np.clip(data_array['res'],0.7,5.0)
          try:
               self.jsu_coeffs
          except:
               self.read_jsu_coeff()
          #initialize arrays
          #for each feature, likelihood values from jsu distributions
          likelihood_s = np.zeros((data_array.shape[0],len(selectors.pca_view_col)))
          likelihood_w = np.zeros(likelihood_s.shape)
          #baselines are for calibration of LLG, data is centered with the average
          #between s and w populations set to zero, so inputting zero as observations
          #estimates baseline likelihood given 50/50 probability (I think)
          baseline_rand = np.zeros(likelihood_s.shape)
          dev_s = np.zeros(likelihood_s.shape)
          dev_w = np.zeros(likelihood_s.shape)
          jsu_mean_s = np.zeros(likelihood_s.shape)
          jsu_mean_w = np.zeros(likelihood_s.shape)
          jsu_var_s = np.zeros(likelihood_s.shape)
          jsu_var_w = np.zeros(likelihood_s.shape)
          #iterate by feature (column) -- bit clumsy
          for index,column in enumerate(selectors.pca_view_col):
               s_pdf_coeff,w_pdf_coeff=self.get_jsu_coeffs(index,resolution)
               likelihood_s[:,index] = self.johnsonsu_pdf(data_array[column],*s_pdf_coeff)
               likelihood_w[:,index] = self.johnsonsu_pdf(data_array[column],*w_pdf_coeff)
               jsu_mean_s[:,index],jsu_var_s[:,index] = self.johnsonsu_stats(s_pdf_coeff)
               jsu_mean_w[:,index],jsu_var_w[:,index] = self.johnsonsu_stats(w_pdf_coeff)
               baseline_rand[:,index] = self.norm_pdf(data_array[column])
               #store deviations from distribution means
               dev_s[:,index] = np.subtract(data_array[column],jsu_mean_s[:,index])
               dev_w[:,index] = np.subtract(data_array[column],jsu_mean_w[:,index])


          #clip likelihoods to avoid underrun and artifacts from imperfect distributions (in place)
          np.clip(likelihood_s,0.0001,np.inf,out=likelihood_s)
          np.clip(likelihood_w,0.0001,np.inf,out=likelihood_w)
          np.clip(jsu_var_s,0.001,np.inf,out=jsu_var_s)
          np.clip(jsu_var_w,0.001,np.inf,out=jsu_var_w)
          #linear ind likelihoods, sum logs for total
          ll_s = np.nansum(np.log(likelihood_s),axis=1)
          ll_w = np.nansum(np.log(likelihood_w),axis=1)
          ll_rand = np.nansum(np.log(baseline_rand),axis=1)
          #added to give log likelihood gain LLG
          llg_s = ll_s - ll_rand
          llg_w = ll_w - ll_rand
          llg_ratio = np.subtract(llg_s,llg_w)
          #chisq calculations, inline multiplication faster
          chisq_s = np.nansum(np.divide(np.multiply(dev_s,dev_s),jsu_var_s),axis=1)
          chisq_w = np.nansum(np.divide(np.multiply(dev_w,dev_w),jsu_var_w),axis=1)
          #store score as LLG ratio 
          #write to pre-instantiated structured array
          results_array['id'] = data_array['id']
          results_array['res'] = data_array['res']
          results_array['score'] = llg_ratio
          results_array['prob'] = self.llr_to_prob(llg_ratio)
          results_array['llgS'] = llg_s
          results_array['llgW'] = llg_w
          results_array['chiS'] = chisq_s
          results_array['chiW'] = chisq_w

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

     def score_stats(self,data_array,results_array,plot=False):
          selectors = Selectors(data_array)
          if plot:
               gridplot = plt.figure(figsize=(12,12))
               plot_data = []
          self.score_breakdown(data_array,results_array)
          result_class = results_array['rc']

          for resbin in range(10):
               if resbin == 0: #all data
                    selected_data = data_array
                    selected_results = results_array
                    selected_class = result_class
               else:
                    selected_data = data_array[data_array['bin'] == resbin]
                    selected_results = results_array[data_array['bin'] == resbin]
                    selected_class = result_class[data_array['bin'] == resbin]
               count = selected_data.shape[0]
               if plot and count < 10: #not enough data in bin
                    plot_data.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    break
               counts = []
               for rclass in np.arange(1,25,1):
                    counts.append(np.count_nonzero(selected_class == rclass))

               
               print "STATS FOR BIN %1s PEAKS %8s" % (resbin,count)
               for index,s_class in enumerate(('ALL_DAT','POS_LLG')):
                    res_arr =np.array(counts).reshape((6,4)) 
                    if index == 0:
                         tp,tn,fp,fn = np.nansum(res_arr[0:4,:],axis=1)
                         count = np.nansum(res_arr[0:4,:])
                         other_s,other_w = np.nansum(res_arr[4,:]),np.nansum(res_arr[5,:])
                         otot = other_s + other_w
                    else:
                         #remove classes with bad scores for both 4,8,12,16,20,24            
                         res_arr[:,-1] = 0
                         tp,tn,fp,fn = np.nansum(res_arr[0:4,:],axis=1) 
                         count = np.nansum(res_arr[0:4,:])
                         other_w,other_s = np.nansum(res_arr[4,:]),np.nansum(res_arr[5,:])
                         otot = other_s + other_w
                    acc,ppv,npv,rec,f1 = self.score_class(tp,tn,fp,fn)
                    count = np.clip(count,1,np.inf)
                    otot = np.clip(otot,1,np.inf)
                    ftp,ftn,ffp,ffn = np.array((tp,tn,fp,fn))/float(count)
                    print "    %s   TP %5s(%4.3f) TN %5s(%4.3f) FP %5s(%4.3f) FN %5s(%4.3f)" % (s_class,tp,ftp,tn,ftn,fp,ffp,fn,ffn)
                    print "      No SW Label: Total %5d PredS %5s(%4.3f) PredW %5s(%4.3f)" % (otot,other_s,float(other_s)/otot,other_w,
                                                                                              float(other_w)/otot)
                    print "      RC: ", " ".join('{:5d}'.format(x) for x in np.arange(1,13,1))
                    print "          ", " ".join('{:5d}'.format(x) for x in res_arr.reshape(-1)[0:12])
                    print "      RC: ", " ".join('{:5d}'.format(x) for x in np.arange(13,25,1))
                    print "          ", " ".join('{:5d}'.format(x) for x in res_arr.reshape(-1)[12:25])
                    print "     RATIOS: ACC %4.3f    PPV %4.3f    NPV %4.3f    REC %4.3f        F1 %4.3f" % (acc,ppv,npv,rec,f1)
                    print ""

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

                   
     def peak_plot(self,peak_data):
          plt.switch_backend('WX')
          resolution = peak_data['res']

          db_id = peak_data['id']
          pca_view_col = ['RX0','RX1','RX2','RX3','RX4','RX5',
                          'RX6','RX7','RX8','RX9','RX10','RX11',
                          'RX12','RX13','RX14','RX15','RX16','RX17','RX18']
          seldata = np.array(peak_data.reshape(-1)[pca_view_col])
          gridplot = plt.figure(figsize=(24,8))  
          xdata=np.linspace(-5,5,200)
          for index,column in enumerate(pca_view_col):
               sub = gridplot.add_subplot(4,5,index+1)
               data_point = seldata[column]
               sub.plot((data_point,data_point),(0.0,1.0),'k-')
               sfit,wfit = self.get_jsu_coeffs(index,resolution)
               sub.plot(xdata,self.johnsonsu_pdf(xdata,sfit[0],sfit[1],sfit[2],sfit[3]),'r-')
               sub.plot(xdata,self.johnsonsu_pdf(xdata,wfit[0],wfit[1],wfit[2],wfit[3]),'b-')
          #plt.show()
          plt.savefig("POINT_FIT_"+db_id+".png")
          plt.clf()
          plt.close()


     def peak_edc(self,results_array):
          #function to assign electron density class
          #1-8, 1=best s, 8=best water


          #cutoffs for "good" values for llg and chisq (empirical)
          #cutoffs for score/chi S/W
          gss_cut = -3.0
          gsc_cut = 65
          gws_cut = -3.0
          gwc_cut = 55

          preds = results_array['score'] > 0.0
          goods_ss = results_array['llgS'] > gss_cut
          goods_sc = results_array['chiS'] < gsc_cut
          ps_gs = np.logical_and(preds,goods_ss)
          ps_gc = np.logical_and(preds,goods_sc)
          
          predw = np.invert(preds)
          goodw_ws = results_array['llgW'] > gws_cut
          goodw_wc = results_array['chiW'] < gwc_cut
          pw_gs = np.logical_and(predw,goodw_ws)
          pw_gc = np.logical_and(predw,goodw_wc)

          good_score = np.logical_or(ps_gs,pw_gs)
          good_chi = np.logical_or(ps_gc,pw_gc)

          logical_ass = np.zeros((results_array.shape[0],3),dtype=np.bool_)
          logical_ass[:,0] = preds
          logical_ass[:,1] = good_score
          logical_ass[:,2] = good_chi

          #explicit def of all edc
          edc = np.zeros(results_array.shape[0],dtype = np.int16)
          edc1 =(logical_ass == (1,1,1)).all(axis=1) #preds,goods,goodc --> good S
          edc2 =(logical_ass == (1,0,1)).all(axis=1) #preds,bads,goodc --> weak S
          edc3 =(logical_ass == (1,1,0)).all(axis=1) #preds,goods,badc --> bad S
          edc4 =(logical_ass == (1,0,0)).all(axis=1) #preds,bads,badc --> really bad S
          edc5 =(logical_ass == (0,0,0)).all(axis=1) #predw,bads,badc --> really bad W
          edc6 =(logical_ass == (0,1,0)).all(axis=1) #predw,goods,badc --> bad W
          edc7 =(logical_ass == (0,0,1)).all(axis=1) #predw,bads,goodc --> weak W
          edc8 =(logical_ass == (0,1,1)).all(axis=1) #predw,goods,goodc --> good W

          for edc_ind,selector in enumerate((edc1,edc2,edc3,edc4,
                                             edc5,edc6,edc7,edc8)):
               edc[selector] = edc_ind + 1

          return edc

     def peak_fc(self,cfeat_array):
          #function to assign a class for various peak flags
          # 0 = no flags
          # 1 = special position
          # 2 = very bad contacts
          # 3 = bad contacts and one close contact
          # 4 = bad contacts
          # 5 = one close contact
          # 6 = remote, far from any contact

          flags_col=['weak','remote','close','special','badc','sadc']
          flags_fmt = [np.bool_,np.bool_,np.bool_,np.bool_,np.bool_,np.bool_]
          flags_dtype = np.dtype(zip(flags_col,flags_fmt))
          flags_arr = cfeat_array[flags_col].view(dtype=flags_dtype)

          flag_class = np.zeros(flags_arr.shape[0],dtype=np.int16)
          sp_sel = flags_arr['special'] == True
          badc_sel = flags_arr['badc'] == True
          sadc_sel = flags_arr['sadc'] == True
          close_sel = flags_arr['close'] == True
          remote_sel = flags_arr['remote'] == True

          #assign in rev order of precidence (remote cannot be special, special implies badc, etc.)
          flag_class[remote_sel] = 6
          flag_class[close_sel] = 5
          flag_class[sadc_sel] = 4
          flag_class[np.logical_and(sadc_sel,close_sel)] = 3
          flag_class[badc_sel] = 2
          flag_class[sp_sel] = 1


          return flag_class

     def peak_cc(self,results_array,cfeat_array):
          #function to give a class for local contact environment
          swt_cut = 0 # max contact close contact counts for refined w
          wwt_cut = 1
          sst_cut = 3
          scl_cut = 2.8 #closest contact
          wcl_cut = 2.0
          #total short contacts to refined water position
          wt = cfeat_array['wl'] + cfeat_array['wm'] 
          st = cfeat_array['sl'] + cfeat_array['sm'] 

          #probability of SO4 via contact data Random Forest          
          cpred_s = cfeat_array['cprob'] > 0.8
          cpred_w = np.invert(cpred_s)
          goods_c1 = cfeat_array['c1'] > scl_cut
          goodw_c1 = cfeat_array['c1'] > wcl_cut
          goods_wt = wt <= swt_cut
          goodw_wt = wt <= swt_cut

          goods_wt = np.logical_and(cpred_s,wt <= swt_cut) #good s cont
          goods_st = np.logical_and(cpred_s,st <= sst_cut)
          goods_tc = np.logical_and(goods_wt,goods_st)
          goodw_wt = np.logical_and(cpred_w,wt <= wwt_cut) #good w cont
          goods_cl = np.logical_and(cpred_s,cfeat_array['c1'] > scl_cut) #good close s cont
          goodw_cl = np.logical_and(cpred_w,cfeat_array['c1'] > wcl_cut) #good close w cont
          good_wt = np.logical_or(goods_tc,goodw_wt)
          good_cl = np.logical_or(goods_cl,goodw_cl)


          lcprob = cfeat_array['cprob']

          #logical_ass = np.zeros((results_array.shape[0],3),dtype=np.bool_)
          #logical_ass[:,0] = cpred_s #contact prob
          #logical_ass[:,1] = good_wt #good wt score for assigned class
          #logical_ass[:,2] = good_cl #good close contact for assigned class

          c_class = np.zeros(results_array.shape[0],dtype=np.int16)
          #cc1 =(logical_ass == (1,1,1)).all(axis=1) #cpreds,good, good 
          #cc2 =(logical_ass == (1,1,0)).all(axis=1) #cpreds, good, bad close
          #cc3 =(logical_ass == (1,0,1)).all(axis=1) #cpreds, bad contact env, good close
          #cc4 =(logical_ass == (1,0,0)).all(axis=1) #cpreds, bad contacts
          #cc5 =(logical_ass == (0,1,1)).all(axis=1) #cpredw bad all round
          #cc6 =(logical_ass == (0,0,1)).all(axis=1) #cpredw,bad env
          #cc7 =(logical_ass == (0,1,0)).all(axis=1) #cpredw, bad close contact
          #cc8 =(logical_ass == (0,1,1)).all(axis=1) #cpredw,good contacts

          cc1 = lcprob < 1.01
          cc2 = lcprob < 0.99
          cc3 = lcprob < 0.98
          cc4 = lcprob < 0.90
          cc5 = lcprob < 0.80
          cc6 = lcprob < 0.60
          cc7 = lcprob < 0.40
          cc8 = lcprob < 0.20





          for cc_ind,selector in enumerate((cc1,cc2,cc3,cc4,
                                            cc5,cc6,cc7,cc8)):
               c_class[selector] = cc_ind + 1

          return c_class


     def peak_sfp(self,psfp_array):
          #looks at two random forest classifiers
          #for false positives, one on density, one on contacts

          pos_cpsfp = psfp_array['cpsfp'] > 0.65
          pos_epsfp = psfp_array['epsfp'] > 0.40
          comp_psfp = np.multiply(psfp_array['cpsfp'],psfp_array['epsfp'])
          perfect_comp = comp_psfp == 1.0 # perfect score
          pos_comp = comp_psfp > 0.95 # very good composite score

          fp_lc = np.zeros((psfp_array.shape[0],7),dtype=np.bool_)
          fp_lc[:,0] = comp_psfp == 1.0 #both fp scores = 1.0
          fp_lc[:,1] = psfp_array['epsfp'] == 1.0
          fp_lc[:,2] = psfp_array['cpsfp'] == 1.0
          fp_lc[:,3] = comp_psfp > 0.8
          fp_lc[:,4] = psfp_array['cpsfp'] < 0.5
          fp_lc[:,5] = psfp_array['cpsfp'] < 0.2
          fp_lc[:,6] =  psfp_array['cpsfp'] == 0


          fp_class = np.zeros(psfp_array.shape[0],dtype=np.int16) + 5 #assign default of 5
          fpc1 =(fp_lc == (1,1,1,1,0,0,0)).all(axis=1) #perfect TP score
          fpc2 =(fp_lc == (0,0,1,1,0,0,0)).all(axis=1) #only perfect efp score
          fpc3 =(fp_lc == (0,1,0,1,0,0,0)).all(axis=1) #only perfect cfp score
          fpc4 =(fp_lc == (0,0,0,1,0,0,0)).all(axis=1) #good composite score
          fpc5 =(fp_lc == (0,0,0,0,1,1,0)).all(axis=1) #catchall -- neither
          fpc6 =(fp_lc == (0,0,0,0,1,0,0)).all(axis=1) #low csfp score
          fpc7 =(fp_lc == (0,0,0,0,1,1,0)).all(axis=1) #very low csfp score
          fpc8 =(fp_lc == (0,0,0,0,1,1,1)).all(axis=1) #perfect FP score



          for fpc_ind,selector in enumerate((fpc1,fpc2,fpc3,fpc4,fpc5,fpc6,fpc7,fpc8)):
               fp_class[selector] = fpc_ind + 1


          #total = 0
          #for i in range(9):
          #     print "FP CLASS COUNT",i,np.count_nonzero(fp_class == i)
          #     total = total + np.count_nonzero(fp_class == i)
          #print "TOTAL",total,psfp_array.shape[0]


          return fp_class


