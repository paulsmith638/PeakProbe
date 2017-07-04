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
          print "     PCA XFORM MATRIX RES %.2f DET %.2f" % (res,np.linalg.det(xform_matrix))
          return xform_matrix


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
               #new array scores pass to original
               data_array['score'][sel_ind] = input_results['score']
          return results_array

     def initialize_results(self,data_array):
          rows = data_array.shape[0]
          dtype = [('id','S16'),('res','f4'),('score','f4'),('prob','f4'),
                   ('llgS','f4'),('llgW','f4'),('chiS','f4'),('chiW','f4'),
                   ('fchi','f4'),('kchi','f4')]
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
          data_array['score'] = llg_ratio
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
          #2 = obs S
          #3 = good S score
          #4 = good W score
          logical_ass = np.zeros((data_array.shape[0],4),dtype=np.bool_)
          # returns boolean masks array
          selectors = Selectors(data_array)
          logical_ass[:,0] = selectors.inc_obss_bool
          logical_ass[:,1] = results_array['score'] >= 0 
          logical_ass[:,2] = results_array['llgS'] > 0.0
          logical_ass[:,3] = results_array['llgW'] > 0.0

          lc1110 = (logical_ass == (1,1,1,0)).all(axis=1) #1 = TP w / bad water score
          lc1101 = (logical_ass == (1,1,0,1)).all(axis=1) #2 = impossible
          lc1111 = (logical_ass == (1,1,1,1)).all(axis=1) #3 = TP w / good water score
          lc1100 = (logical_ass == (1,1,0,0)).all(axis=1) #4 = TP with bad scores

          lc0010 = (logical_ass == (0,0,1,0)).all(axis=1) #5 = impossible
          lc0001 = (logical_ass == (0,0,0,1)).all(axis=1) #6 = TN with bad S score
          lc0011 = (logical_ass == (0,0,1,1)).all(axis=1) #7 = TN with good S score
          lc0000 = (logical_ass == (0,0,0,0)).all(axis=1) #8 = TN with bad scores

          lc0110 = (logical_ass == (0,1,1,0)).all(axis=1) #9 = FP with good S score (bad label?)
          lc0101 = (logical_ass == (0,1,0,1)).all(axis=1) #10 = impossible
          lc0111 = (logical_ass == (0,1,1,1)).all(axis=1) #11 = FP with good W score
          lc0100 = (logical_ass == (0,1,0,0)).all(axis=1) #12 = FP with bad scores

          lc1010 = (logical_ass == (1,0,1,0)).all(axis=1) #13 = impossible
          lc1001 = (logical_ass == (1,0,0,1)).all(axis=1) #14 = FN with good W score (bad label?)
          lc1011 = (logical_ass == (1,0,1,1)).all(axis=1) #15 = FN with good S score
          lc1000 = (logical_ass == (1,0,0,0)).all(axis=1) #16 = FN with bad scores

          result_class = np.zeros(data_array.shape[0],dtype=np.int16)
          for index,lclass in enumerate((lc1110,lc1101,lc1111,lc1100,lc0010,lc0001,lc0011,lc0000,
                                         lc0110,lc0101,lc0111,lc0100,lc1010,lc1001,lc1011,lc1000)):
               select = lclass
               result_class[select] = index + 1
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
          if (numTP+numFP+numFN) > 0:
               f1 = (2.0*numTP)/(2.0*numTP+numFP+numFN)#harmonic mean of ppv and rec
          else:
               f1 = 0.0
          return acc,ppv,npv,rec,f1

     def score_stats(self,data_array,results_array,plot=False):
          selectors = Selectors(data_array)
          if plot:
               gridplot = plt.figure(figsize=(12,12))
               plot_data = []
          result_class = self.score_breakdown(data_array,results_array)

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
               if count < 10: #not enough data in bin
                    plot_data.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    break
               tpsel = np.logical_and(selected_class > 0,selected_class <=4)
               tnsel = np.logical_and(selected_class > 4,selected_class <=8)
               fpsel = np.logical_and(selected_class > 8,selected_class <=12)
               fnsel = np.logical_and(selected_class > 12,selected_class <=16)
               counts = []
               for rclass in np.arange(1,17,1):
                    counts.append(np.count_nonzero(selected_class == rclass))

               
               print "STATS FOR BIN %1s PEAKS %8s" % (resbin,count)
               tp,tn,fp,fn = np.nansum(np.array(counts).reshape((4,4)),axis=1)
               acc,ppv,npv,rec,f1 = self.score_class(tp,tn,fp,fn)
               ftp,ftn,ffp,ffn = np.array((tp,tn,fp,fn))/float(count)
               print "    ALL DATA   TP %5s(%4.3f) TN %5s(%4.3f) FP %5s(%4.3f) FN %5s(%4.3f)" % (tp,ftp,tn,ftn,fp,ffp,fn,ffn)
               print "               ACC %4.3f       PPV %4.3f       NPV %4.3f       REC %4.3f        F1 %4.3f" % (acc,ppv,npv,rec,f1)
               print "        RC: ", "".join('{:5d}'.format(x) for x in np.arange(1,17,1))
               print "            ", "".join('{:5d}'.format(x) for x in counts)
               print ""
               if plot:
                    plot_data.append(counts)
          if plot:
               plot_data = np.array(plot_data)
               bincounts = np.nansum(plot_data,axis=1)
               nplot_data = np.divide(plot_data,bincounts[:,None])
               for rclass in np.arange(1,17,1):
                    sub = gridplot.add_subplot(4,4,rclass)
                    bar_data = nplot_data[:,rclass-1]
                    sub.bar(np.arange(10),bar_data,align='center',alpha=0.5)
                    sub.set_xticks(np.arange(10))
                    sub.set_xticklabels(np.arange(10))
                    sub.set_title("RC_"+str(rclass))
               plt.savefig("SCORE_DIST.png")
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



