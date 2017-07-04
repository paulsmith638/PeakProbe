import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from PProbe_selectors import Selectors
from PProbe_stats import StatFunc

class SynData:
    def __init__(self,rseed=0):
        np.random.seed(rseed)
        self.ppsel = Selectors(None)
        self.ppstat = StatFunc()

    def gen_syn_data(self,size,datatype='pca',sfrac = 0.50,plot=False):
        if datatype == 'raw':
            colnames = self.ppsel.alldata_input_col
            col_dtype = self.ppsel.alldata_input_dtype
            feature_col = self.ppsel.std_view_col
        if datatype == 'pca':
            colnames = self.ppsel.alldata_pca_col
            col_dtype = self.ppsel.alldata_pca_dtype
            feature_col = self.ppsel.pca_view_col
        #initialize structured array of appropriate dtype
        synth_data = np.zeros(size,dtype=col_dtype)
        num_data_cols = len(colnames)#usually 19, may change
        #randomly assign some to be sulfate
        fracs = sfrac
        fracw = 1.0 - fracs
        syn_so4_mask = np.array(np.random.choice((0,1),p=(fracw,fracs),size=size),dtype=np.bool)
        synth_data['ori'][syn_so4_mask]='SO4'
        synth_data['ori'][np.invert(syn_so4_mask)]='HOH'
        #uniform resolutions across all observed range
        synth_data['res'] = np.random.uniform(0.65,5.1,size)
        synth_data['id'][syn_so4_mask] = "rand_S_00000"
        synth_data['id'][np.invert(syn_so4_mask)] = "rand_W_00000"
        synth_data['bin'] = self.ppstat.assign_bin(synth_data['res'])
        synth_data['batch'] = np.random.random_integers(0,999,size)#inclusive function
        synth_data['omit'] = False
        synth_data['solc'] = np.clip(np.random.normal(loc=0.5,scale=0.15,size=size),0.2,0.8)
        synth_data['fofc_sigi'] = np.random.normal(loc=4.0,scale=1.0,size=size)
        synth_data['2fofc_sigi'] = np.random.normal(loc=4.0,scale=1.0,size=size)
        synth_data['fofc_sigo'] = np.random.normal(loc=4.0,scale=1.0,size=size)
        synth_data['2fofc_sigo'] = np.random.normal(loc=4.0,scale=1.0,size=size)
        synth_data['dmove'] = np.random.normal(loc=1.0,scale=0.3,size=size)
        #generate feature data, some random means and std
        #two populations, seperation is random, but resembles real data 
        random_means=np.random.uniform(-1.5,1.5,num_data_cols*2)
        random_std=np.random.uniform(0.4,1.5,num_data_cols*2)
        if plot:
            gridplot = plt.figure(figsize=(16,12))
        for index,column in enumerate(feature_col):
            s_mean = random_means[index]
            w_mean = random_means[index+num_data_cols]
            s_std = random_std[index]
            w_std = random_std[index + num_data_cols]
            #generate histogram to pick values from, covers 10 units on x-axis
            s_val,s_prob = self.noisy_gaussian(s_mean,s_std,1000)
            w_val,w_prob = self.noisy_gaussian(w_mean,w_std,1000)
            ns = np.count_nonzero(syn_so4_mask)
            nw = size - ns
            s_data = np.random.choice(s_val,p=s_prob,size=ns)
            w_data = np.random.choice(w_val,p=w_prob,size=nw)


            #introduct wobble to keep data from spiking as values are drawn
            #from histogram (discrete), bin witdh = 0.01
            s_wobb = np.random.uniform(-0.005,0.005,size=ns)
            w_wobb = np.random.uniform(-0.005,0.005,size=nw)
            synth_data[column][syn_so4_mask]=np.add(s_data,s_wobb)
            synth_data[column][np.invert(syn_so4_mask)]=np.add(w_data,w_wobb)
            if plot:
                sub = gridplot.add_subplot(4,5,index+1)
                xlow = np.amin(np.concatenate((s_val,w_val)))
                xhigh = np.amax(np.concatenate((s_val,w_val)))
                sub.hist(s_data, normed=True, bins=50,range=(xlow,xhigh),color="red")
                sub.hist(w_data, normed=True, bins=50,range=(xlow,xhigh),color="blue")
        #add resdep correlation for raw data
        if datatype == 'raw':
            print "INPUT COV"
            print np.cov(synth_data[feature_col].view(self.ppsel.raw_dtype).T)
            corr_perturb = np.zeros((size,len(feature_col)))
            for resbin in np.arange(1,10,1):
                binsel = synth_data['bin'] == resbin
                binsize = np.count_nonzero(binsel)
                means=np.zeros(len(feature_col))
                covin = np.zeros((len(feature_col),len(feature_col)))
                for i in range(covin.shape[0]):
                    for j in range(covin.shape[1]):
                        if i == j:
                            covin[i,j] = 1.0
                        if i < j:
                            covin[i,j] = np.random.random()*np.sin(resbin)**2
                        if i > j:
                            covin[i,j] = covin[j,i]
                cdata = np.random.multivariate_normal(means, covin, binsize)*0.5
                corr_perturb[binsel] = cdata
            for index,column in enumerate(feature_col):
                synth_data[column] = synth_data[column] + corr_perturb[:,index]
            print "OUTPUT COV"
            print np.cov(synth_data[feature_col].view(self.ppsel.raw_dtype).T)




            
        if plot:
            plt.savefig("RANDOM_DATA_"+datatype+".png")
            plt.close()
        return synth_data

    def noisy_gaussian(self,loc,scale,size):
        #NB, returns a distribution and associated x-values to be used as a distribution
        xmin = -5 + loc
        xmax = 5 + loc
        xrange = np.linspace(xmin,xmax,size)
        norm_pdf = lambda x,loc,scale: (1.0/np.sqrt(2.0*np.pi*scale*scale))*np.exp(-((x-loc)*(x-loc))/(2.0*scale*scale))
        gaussf = norm_pdf(xrange,loc,scale)
        gmax = norm_pdf(loc,loc,scale)
        bit_of_noise = np.random.uniform(low=-0.2*gmax,high=0.2*gmax,size=size)
        noise_sinc_scale =np.sinc(xrange-loc)
        total_noise = np.multiply(noise_sinc_scale,bit_of_noise)
        skew = np.random.uniform(-0.05,0.05,size=1)*(xrange-xmin) + 0.95
        gn = np.clip(np.add(total_noise,gaussf)*skew,0.001,np.inf)
        gnsum = np.nansum(gn)
        gnorm = gn/gnsum
        #plt.plot(xrange,gaussf,color='r')
        #plt.plot(xrange,skew,color='b')
        #plt.plot(xrange,gn,color='m')
        #plt.show()
        return xrange,gnorm


    def initialize_results(self,data_array):
        rows = data_array.shape[0]
        dtype = [('id','S16'),('res','f4'),('score','f4'),('prob','f4'),
                 ('llgS','f4'),('llgW','f4'),('chiS','f4'),('chiW','f4'),
                 ('fchi','f4'),('kchi','f4')]
        return np.zeros(rows,dtype=dtype)



    def gen_syn_results(self,size):
        results_array = np.zeros(size)#dummy
        results_array = self.initialize_results(results_array)
        results_array['id'] = "rrnd_X_00000"
        results_array['res']= np.random.uniform(0.7,4.8,size)
        #create two pseudo populations for likelihood functions
        random_means=np.random.uniform(-20.0,20.0,4)
        random_std=np.random.uniform(10,20,4)
        pop1 = np.random.normal(loc=random_means[0],scale=random_std[0],size=size)
        pop2 = np.random.normal(loc=random_means[1],scale=random_std[1],size=size)
        pop3 = np.random.normal(loc=random_means[2],scale=random_std[2],size=size)
        pop4 = np.random.normal(loc=random_means[3],scale=random_std[3],size=size)
        jpop1 = np.add(pop1,pop2)
        jpop2 = np.add(pop3,pop4)
        results_array['llgS'] = jpop1
        results_array['llgW'] = jpop2
        results_array['score'] = np.subtract(jpop1,jpop2)
        results_array['prob'] = 1.0/(np.exp(-results_array['score']) + 1.0)  
        ripple = np.sinc(np.linspace(0,size,size) - size/5)
        bit_of_noise = np.random.uniform(low=-0.3,high=0.3,size=size)
        total_noise = np.multiply(ripple,bit_of_noise)
        results_array['chiS'] = np.random.chisquare(19,size=size)+total_noise
        results_array['chiW'] = np.random.chisquare(19,size=size)+total_noise[::-1]
        results_array['fchi'] = np.random.chisquare(19,size=size)+total_noise
        results_array['kchi'] = np.random.chisquare(3,size=size)+total_noise[::-1]
        return results_array
                           
          




    def randomize_data(self,data_array):
        #shuffles labels
        print "RANDOMIZING DATA"
        rand_ori = data_array['ori'].copy()
        random_choice = ['rand_S_00001','rand_W_00001']
        random_data = data_array.copy()
        random_data['ori'] = np.random.shuffle(rand_ori)
        random_data['id'] = np.random.choice(random_choice,size=data_array.shape[0])
        return random_data
