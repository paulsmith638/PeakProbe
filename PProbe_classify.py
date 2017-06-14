import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
#import sqlite3 as lite
from PProbe_selectors import Selectors as ppsel
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
          self.ppsel = ppsel

# Distributions of data in PCA space were skew-normal like, so fit to 
# JSU distribution, which includes analogs of skew and kurtosis and has 
# support for all real numbers

     def johnsonsu_stats(self,param):
          #in wikipedia notation, xi is loc, lambda is scale,delta is b, gamma a
          a,b,loc,scale = param
          xi = loc
          lam = scale
          delta = b
          gamma = a
          pdf_mean = (xi - lam*np.exp((delta**-2)/2)*np.arcsinh(gamma/delta))
          pdf_var = ((lam**2)/2)*(np.exp(delta**-2)-1.0)*(np.exp(delta**-2)*np.cosh((2*gamma)/delta) + 1)
          return pdf_mean,pdf_var

     def johnsonsu_pdf(self,xval,a,b,loc,scale):
          #borrowed from scipy 
          xshift = (xval-loc)/scale
          prefactor1 = b/((scale*np.sqrt(2*np.pi))*(np.sqrt(1+xshift**2)))
          prob = prefactor1*np.exp(-0.5*(a+b*np.arcsinh(xshift))**2)
          return prob

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
          try:
               self.jsu_coeffs
          except:
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


     #@profile
     def gen_xform_mat(self,res):
          #takes a resolution and outputs a matrix
          #of pseudo-eigenvectors generated from spline 
          #coefficients in numpy eigvect,component (row) format
          #only works "one at a time" (slow)
          pca_coeffs = self.resscale_pca_coeffs
          num_col = int(np.sqrt(len(pca_coeffs)))
          #initialze a matrix
          xform_matrix = np.zeros((num_col,num_col))
          for i in range(xform_matrix.shape[0]):
               for j in range(xform_matrix.shape[1]):
                    column_index=str(i)+"_"+str(j)
                    params = pca_coeffs[column_index]
                    xform_matrix[i,j] = self.spline4k(params,res.reshape(-1))
          return xform_matrix


     def xform_data(self,data,matrix):
          #matrix of norm eigenvectors passed as rows (numpy normally outputs columns)
          #data passed as rows, is transposed, xform is back-transposed to rows
          return np.dot(matrix,data.T).T


     def standardize_data(self,data_array,post_pca=False):
          """
          centers data by subtracting the resolution dependent s/w mean and sigma
          centering puts zero as the midpoint between sulfate and water means
          """
          selectors=ppsel(data_array)
          #read in the dictionary of spline coefficients for raw data scaling


          #remove peaks not flagged as omit
          data_array = data_array[selectors.included_data_bool]
          scaled_array=data_array.copy()
          resolution_column = data_array['res']
          #method works with data both before and after PCA transformation
          #setup accordingly with array column names and scale coeff dictionaries
          if post_pca:
               column_list = selectors.pca_view_col
               try:
                    scale_post_pca_file = open("pprobe_post_pca_resscales.dict",'r')
               except:
                    sys.exit("CANNOT OPEN FILE: pprobe_post_pca_resscales.dict -- TERMINATING!")
               self.scale_post_pca_spline_coeffs = ast.literal_eval(scale_post_pca_file.read())
               scale_post_pca_file.close()

          else:
               column_list = selectors.std_view_col
               try:
                    scale_raw_feature_file = open("pprobe_pre_pca_resscales.dict",'r')
               except:
                    sys.exit("CANNOT OPEN FILE: pprobe_pre_pca_resscales.dict -- TERMINATING!")
               self.scale_raw_feature_spline_coeffs = ast.literal_eval(scale_raw_feature_file.read())
               scale_raw_feature_file.close()


          for column in column_list:
               calc_meanx,calc_stdev = self.get_stats(data_array[column])
               if self.verbose:
                    print "INPUT COLUMN",column,calc_meanx,calc_stdev
               if post_pca == False:
                    resd_mean,resd_sig = self.get_res_scales(column,resolution_column)
               else:
                    resd_mean,resd_sig = self.get_post_pca_scales(column,resolution_column)

               scaled_array[column] = np.divide(np.subtract(scaled_array[column],resd_mean),np.exp(resd_sig))

               #check normalization applied correctly
               calc_meanx,calc_stdev = self.get_stats(scaled_array[column])
               if self.verbose:
                    print "OUTPUT COLUMN",column,calc_meanx,calc_stdev
          print "SCALED %s ROWS DATA" % scaled_array.shape[0]
          return scaled_array


     def pca_xform_data(self,norm_data,plot=False):
          """
          carries out PCA transformation, first generates matrix for PCA transformation given
          an input resolution, then applies it to one or more data points
          """
          
          print "PCA TRANSFORMATION"
          selectors=ppsel(norm_data)
          num_data = norm_data[selectors.std_view_col].view(selectors.raw_dtype)
          res = norm_data['res']
          info_columns = ['ori','res','id','bin','batch','omit','solc','fofc_sigi','2fofc_sigi','fofc_sigo','2fofc_sigo','dmove','score']
          info_formats = ('S16',np.float64,'S16',np.int16,np.int16,'S32',
                          np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64)
          info_dtype = np.dtype(zip(info_columns,info_formats))
          info_data = norm_data[info_columns].view(info_dtype)
          xformed_data = []
          info_array = []
          #read in PCA transformation matrix coefficients
          #each entry is an i,j matrix entry with 6 spline coefficients used to calculate the eigenvector component at input resolution          
          try:
               pca_matrix_coeff_file = open("pprobe_pca_matrix_coeff.dict",'r')
          except:
               sys.exit("CANNOT OPEN FILE: pprobe_pca_matrix_coeff.dict -- TERMINATING!")
          self.resscale_pca_coeffs = ast.literal_eval(pca_matrix_coeff_file.read())
          pca_matrix_coeff_file.close()

          #fetch a matrix for the 1st peak
          mat = self.gen_xform_mat(res[0])
          for index,row in enumerate(num_data):
               #update matrix if resolution is more than 0.05A different (speedup)
               #matrix coefficients are smoothly varying
               if np.abs(res[index]-res[index-1]) > 0.05: 
                    mat = self.gen_xform_mat(res[index])
               xformed_data.append(tuple(self.xform_data(row,mat)))
               info_array.append(tuple(info_data[index]))
          array_size = len(xformed_data)
          print "TRANSFORMED %s ROWS DATA" % array_size
          pca_array = np.fromiter(((xformed_data[i]+info_array[i]) for i in np.arange(array_size)) ,dtype=selectors.alldata_pca_dtype,count=array_size)
          return pca_array


     def read_jsu_coeff(self):
          #read in JSU coefficient dictionary
          try:
               jsu_coeff_file = open("pprobe_jsu_coeffs.dict",'r')
          except:
               sys.exit("CANNOT OPEN FILE: pprobe_jsu_coeffs.dict -- TERMINATING!")
          self.jsu_coeffs = ast.literal_eval(jsu_coeff_file.read())
          jsu_coeff_file.close() 

     def discriminant_analysis(self,data_array,plot=False):
          print "SCORING %s PEAKS" % data_array.shape[0]
          selectors = ppsel(data_array)
          resolution = data_array['res']
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
          baseline_s = np.zeros(likelihood_s.shape)
          baseline_w = np.zeros(likelihood_s.shape)
          #iterate by feature (column) -- bit clumsy
          for index,column in enumerate(selectors.pca_view_col):
               s_pdf_coeff,w_pdf_coeff=self.get_jsu_coeffs(index,resolution)
               likelihood_s[:,index] = self.johnsonsu_pdf(data_array[column],*s_pdf_coeff)
               likelihood_w[:,index] = self.johnsonsu_pdf(data_array[column],*w_pdf_coeff)
               baseline_s[:,index] = (self.johnsonsu_pdf(np.zeros(data_array.shape[0]),*s_pdf_coeff))
               baseline_w[:,index] = (self.johnsonsu_pdf(np.zeros(data_array.shape[0]),*w_pdf_coeff))
          #clip likelihoods to avoid underrun and artifacts from imperfect distributions (in place)
          np.clip(likelihood_s,0.01,np.inf,out=likelihood_s)
          np.clip(likelihood_w,0.01,np.inf,out=likelihood_w)
          #linear ind likelihoods, sum logs for total
          #ind_llg_s = np.subtract(np.log(likelihood_s),np.log(baseline_s))
          #ind_llg_w = np.subtract(np.log(likelihood_w),np.log(baseline_w))
          #ind_llg_st = np.nansum(ind_llg_s,axis=1)
          #ind_llg_wt = np.nansum(ind_llg_w,axis=1)
          ll_s = np.nansum(np.log(likelihood_s),axis=1)
          ll_w = np.nansum(np.log(likelihood_w),axis=1)
          ll_s_zero = np.nansum(np.log(baseline_s),axis=1)
          ll_w_zero = np.nansum(np.log(baseline_w),axis=1)
          #for index in np.arange(0,data_array.shape[0],1000):
          #     print ll_s[index],ll_w[index],ll_s_zero[index],ll_w_zero[index],ind_llg_st[index],ind_llg_wt[index]
          #added to give log likelihood gain LLG
          llg_s = ll_s - ll_s_zero 
          llg_w = ll_w - ll_w_zero
          llg_ratio = np.subtract(llg_s,llg_w)
          #store score as LLG ratio 
          data_array['score'] = llg_ratio

          #inverse logit to recover probability
          pS = lambda x: 1.0/(np.exp(-x) + 1.0)  

          for index in range(data_array.shape[0]):
               print "PEAK",data_array['id'][index],data_array['res'][index],pS(data_array['score'][index]),data_array['score'][index]
         

     def score_breakdown(self,data_array):
          selectors = ppsel(data_array)
          obss = selectors.inc_obss_bool
          obsw = selectors.inc_obsw_bool
          ress = data_array['score'] >= 0.50
          resw = data_array['score'] < 0.50
          TP = np.logical_and(obss,ress)
          TN = np.logical_and(obsw,resw)
          FP = np.logical_and(obsw,ress)
          FN = np.logical_and(obss,resw)
          return TP,TN,FP,FN


     def score_stats(self,data_array,plot=False,write_res=False):

          if plot:
               gridplot = plt.figure(figsize=(24,8))

          for resbin in range(10):
               if resbin == 0:
                    selected_data = data_array
               else:
                    selected_data = data_array[data_array['bin'] == resbin]
               selectors = ppsel(selected_data)
               count = float(selected_data.shape[0])
               if count == 0:
                    break
               TP,TN,FP,FN = self.score_breakdown(selected_data)
               numTP = np.count_nonzero(TP)
               numTN = np.count_nonzero(TN)
               numFP = np.count_nonzero(FP)
               numFN = np.count_nonzero(FN)
               try:
                    print "SCORE RES BIN %s PEAKS %s TP %s FP %s TN %s FN %s" % (resbin,count,numTP,numFP,numTN,numFN)
                    print "SCORE RES BIN %s PEAKS %s TP %3f FP %3f TN %3f FN %3f" % (resbin,count,numTP/count,numFP/count,
                                                                                     numTN/count,numFN/count)
                    print "     STATS BIN %s ACC %4f PPV %4f NPV %4f REC %4f F1beta %4f" % (resbin,
                                                                                            (numTP + numTN)/count,#accuracy
                                                                                            float(numTP)/(numTP + numFP),#precision
                                                                                            float(numTN)/(numTN + numFN),#neg pred val
                                                                                            float(numTP)/(numTP + numFN),#recall
                                                                                            (2.0*numTP)/(2.0*numTP + numFP + numFN))#F1beta
               except:
                    pass
                    
               if plot:
                    sub = gridplot.add_subplot(2,5,resbin+1)
                    sub.hist(selected_data['score'][selectors.inc_obss_bool], normed=True, bins=30,range=(0.0,1.0),color="red")
                    sub.hist(selected_data['score'][selectors.inc_obsw_bool], normed=True, bins=30,range=(0.0,1.0),color="blue")
          if plot:
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


     def fishers_disc(self,resolution):
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

     def randomize_data(self,data_array):
          #shuffles labels
          print "RANDOMIZING DATA"
          rand_ori = data_array['ori'].copy()
          random_choice = ['rand_S_00001','rand_W_00001']
          random_data = data_array.copy()
          random_data['ori'] = np.random.shuffle(rand_ori)
          random_data['id'] = np.random.choice(random_choice,size=data_array.shape[0])
          return random_data


"""
          so4_probs = np.array(so4_pdf_array).T
          wat_probs = np.array(wat_pdf_array).T
          obs_prob = so4_probs*0.5 + 0.5*wat_probs
          obs_prob = np.dot(np.diag(prior),so4_probs) + np.dot(np.diag((1.0-prior)),wat_probs)
          print "SO4",so4_probs
          print "WAT",wat_probs
          print "OBS",obs_prob
          prod_s = np.prod(so4_probs,axis=1)
          prod_w = np.prod(wat_probs,axis=1)
          prod_o = np.prod(obs_prob,axis=1)
          bayes_s = prod_s/prod_o
          bayes_w = prod_w/prod_o
          bayes = bayes_s/(bayes_s + bayes_w)
          print bayes
"""
