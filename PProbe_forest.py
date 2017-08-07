import os,copy,sys
import numpy as np


class RfContacts:
     """
     class implementing a random forest classifier for contact data
     input data is array whose rows are:
     ['c1','c2','c3','ol','om','sl','sm','wl','wm','ot','st','wt']
     where ot = ol + om, ot = sl + sm and wt = wl + wm
     decision funcions are individual trees from sklearn

     given input data, returns a probability of being a false positive
     based on contact data alone, very useful for finding
     missing ligands as well

     original decision trees (see below) vectorized
     by code also included below
     """
     def __init__(self):
          pass

     def calc_cprob(self,contact_data):
          #p1 p2 are counts from each class in the test set
          #converted to a probability
          p1,p2 = self.contact_tptn_rf(contact_data)

          totals = np.add(p1,p2)
          return np.divide(p1,totals)

     def calc_epsfp(self,results_data):
          #p1 p2 are counts from each class in the test set
          #converted to a probability
          p1,p2 = self.results_tpfp_rf(results_data)

          totals = np.add(p1,p2)
          return np.divide(p1,totals)

     def calc_cpsfp(self,contact_data):
          #p1 p2 are counts from each class in the test set
          #converted to a probability
          p1,p2 = self.contact_tpfp_rf(contact_data)

          totals = np.add(p1,p2)
          return np.divide(p1,totals)


     def union_mask(self,mask_list):
          #something like "reduce" should work, but whatever . . . 
          assert len(mask_list) > 1
          combined_mask = mask_list[0]
          for mask in mask_list[1::]:
               combined_mask = np.logical_and(combined_mask,mask)
          return combined_mask

     def contact_tptn_rf(self,contact_data):
          p1 = np.zeros(contact_data.shape[0])
          p2 = np.zeros(contact_data.shape[0])

          c1 = contact_data['c1']
          c2 = contact_data['c2']
          c3 = contact_data['c3']
          sl = contact_data['sl']
          sm = contact_data['sm']
          st = contact_data['st']
          wt = contact_data['wt']

          sel1 = c1 <= 2.97499990463
          sel2 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.25500011444
          sel3 = c2 <= 3.09499979019
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.14499998093
          sel3 = c2 <= 2.32499980927
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.41499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = c3 <= 3.375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.02500009537
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = st <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.03499984741
          sel2 = c3 <= 3.36499977112
          sel3 = c3 <= 2.40500020981
          p1[self.union_mask([sel1,sel2,sel3])] += 1.0
          p2[self.union_mask([sel1,sel2,sel3])] += 0.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c3 <= 3.46500015259
          sel2 = c2 <= 3.125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = sm <= 0.5
          sel2 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.24499988556
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.08500003815
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.25500011444
          sel3 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = sm <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c2 <= 3.28499984741
          sel2 = c3 <= 3.20499992371
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.44500017166
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.07499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.90500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.27500009537
          sel3 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c2 <= 3.24499988556
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.14499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.06500005722
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.00500011444
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.08500003815
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.42500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 3.10500001907
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.86499977112
          sel3 = c1 <= 1.86500000954
          p1[self.union_mask([sel1,sel2,sel3])] += 1.0
          p2[self.union_mask([sel1,sel2,sel3])] += 0.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.33500003815
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.47499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 2.98500013351
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sl <= 0.5
          sel3 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.19500017166
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.43499994278
          sel2 = c2 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.47499990463
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.47499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.26499986649
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.97499990463
          sel3 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.01499986649
          sel2 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c3 <= 3.54500007629
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.86499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.07499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.07499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.46500015259
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.26499986649
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.42500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c3 <= 3.19500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.02500009537
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.07499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.27500009537
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 3.01499986649
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c1 <= 2.94500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c3 <= 3.20499992371
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.48500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.45499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c3 <= 2.48500013351
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c2 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 3.00500011444
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = c3 <= 3.27500009537
          sel3 = c3 <= 2.44500017166
          p1[self.union_mask([sel1,sel2,sel3])] += 1.0
          p2[self.union_mask([sel1,sel2,sel3])] += 0.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.99499988556
          sel3 = c3 <= 3.27500009537
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.47499990463
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.05499982834
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.22499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c2 <= 3.125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = sl <= 0.5
          sel3 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.45499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c1 <= 3.06500005722
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 3.00500011444
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          sel3 = c1 <= 3.09499979019
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.27500009537
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 1.98500001431
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          sel3 = c2 <= 3.14499998093
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.24499988556
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.41499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c2 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = st <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sl <= 0.5
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.48500013351
          sel3 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 3.50500011444
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.20499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 3.01499986649
          sel2 = st <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c3 <= 3.19500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c3 <= 3.19500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 3.09499979019
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 1.91499996185
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c1 <= 3.04500007629
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.33500003815
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.01499986649
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.18499994278
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 2.99499988556
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.02500009537
          sel2 = c1 <= 2.86499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.23500013351
          sel2 = st <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.02500009537
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c2 <= 3.27500009537
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.09499979019
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c2 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.85500001907
          sel3 = c3 <= 3.42500019073
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 3.01499986649
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.48500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          sel3 = c2 <= 3.26499986649
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.23500013351
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.07499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.99499988556
          sel3 = c1 <= 2.91499996185
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.07499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 1.77499997616
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.09499979019
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          sel2 = c3 <= 3.25500011444
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.22499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = c2 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.85500001907
          sel3 = c3 <= 3.33500003815
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c3 <= 3.46500015259
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.08500003815
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.47499990463
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.00500011444
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 3.00500011444
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.94500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.14499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          sel3 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.28499984741
          sel2 = c3 <= 3.20499992371
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.03499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.46500015259
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          sel3 = c1 <= 2.98500013351
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.27500009537
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.45499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = sm <= 0.5
          sel2 = st <= 0.5
          sel3 = c3 <= 3.35500001907
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c2 <= 3.27500009537
          sel2 = c2 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 3.10500001907
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c3 <= 3.47499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 3.03499984741
          sel2 = c1 <= 2.91499996185
          sel3 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c2 <= 3.24499988556
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.28499984741
          sel2 = c3 <= 3.20499992371
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = c1 <= 2.86499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.00500011444
          sel2 = c2 <= 3.07499980927
          sel3 = c1 <= 1.97500002384
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 3.02500009537
          sel2 = c2 <= 3.15500020981
          sel3 = c2 <= 2.29500007629
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 3.01499986649
          sel2 = c2 <= 3.09499979019
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.47499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 3.17500019073
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.45499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.44500017166
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.98500013351
          sel2 = c2 <= 3.15500020981
          sel3 = c1 <= 1.97500002384
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c3 <= 3.47499990463
          sel2 = c3 <= 3.27500009537
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 1.99500000477
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.14499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.27500009537
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.42500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.51499986649
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          sel2 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.42500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c3 <= 3.36499977112
          sel3 = c1 <= 1.86500000954
          p1[self.union_mask([sel1,sel2,sel3])] += 1.0
          p2[self.union_mask([sel1,sel2,sel3])] += 0.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 3.02500009537
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.46500015259
          sel2 = c1 <= 3.00500011444
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.42500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.86499977112
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c3 <= 3.20499992371
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.06500005722
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.86499977112
          sel3 = c3 <= 3.26499986649
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          sel3 = c2 <= 3.42500019073
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.14499998093
          sel3 = c1 <= 1.91499996185
          p1[self.union_mask([sel1,sel2,sel3])] += 1.0
          p2[self.union_mask([sel1,sel2,sel3])] += 0.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.27500009537
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.49499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.24499988556
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.14499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 3.01499986649
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 3.00500011444
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = st <= 0.5
          sel3 = c1 <= 3.11499977112
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          sel2 = c2 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 3.01499986649
          sel2 = c2 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c3 <= 3.33500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c1 <= 3.03499984741
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.48500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = st <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c3 <= 3.27500009537
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.46500015259
          sel2 = c2 <= 3.18499994278
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.27500009537
          sel2 = c2 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.30499982834
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.10500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.44500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 2.99499988556
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.27500009537
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.48500013351
          sel3 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.39499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 3.02500009537
          sel2 = c3 <= 3.42500019073
          sel3 = c1 <= 2.04500007629
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.08500003815
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 3.01499986649
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.42500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.45499992371
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = st <= 0.5
          sel3 = c2 <= 3.16499996185
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sm <= 0.5
          sel3 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sm <= 0.5
          sel2 = c3 <= 3.42500019073
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0

          return p1,p2


     def results_tpfp_rf(self,results_data):
          p1 = np.zeros(results_data.shape[0])
          p2 = np.zeros(results_data.shape[0])

          score = results_data['score']
          llgS = results_data['llgS']
          llgW = results_data['llgW']
          chiS = results_data['chiS']
          chiW = results_data['chiW']
          sel1 = chiW <= 71.4348449707
          sel2 = chiS <= 10.8713264465
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.4575271606
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.5741500854
          sel2 = llgS <= -3.02684926987
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 8.11484909058
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8550033569
          sel2 = score <= 9.99413299561
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 87.9787979126
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.60970306396
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 41.0577087402
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = chiW <= 200.37197876
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = llgS <= 3.89278101921
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 61.5280609131
          sel3 = score <= 20.4540939331
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 16.5825481415
          sel2 = chiS <= 10.4661140442
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 8.12319469452
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.2691345215
          sel2 = llgS <= -2.71287846565
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9645843506
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.66379499435
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 44.1353530884
          sel3 = score <= 22.7712078094
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 2.71583414078
          sel2 = score <= 9.59057426453
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 14.9965991974
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.7438850403
          sel2 = chiS <= 10.1260051727
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.7287425995
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.5835494995
          sel2 = score <= 9.0055103302
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.8429450989
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.7209205627
          sel2 = llgS <= -3.30973052979
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.822303772
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 68.1023635864
          sel2 = llgS <= 0.63903093338
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 12.9364967346
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8206176758
          sel2 = chiS <= 10.0318088531
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.8146324158
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.68806624413
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 13.0544281006
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = chiS <= 58.3633804321
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = chiW <= 72.5960388184
          sel2 = score <= 9.19959259033
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 44.9445762634
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 72.4402770996
          sel2 = chiS <= 10.1315612793
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.1325855255
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7768707275
          sel2 = chiS <= 10.8739910126
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 19.7231197357
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7567214966
          sel2 = llgS <= -1.42554020882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 9.30447864532
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9168968201
          sel2 = llgS <= 3.13452959061
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 16.3310375214
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6101417542
          sel2 = score <= 8.92564487457
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.7026176453
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.89065885544
          sel2 = score <= 10.1126213074
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 41.9751052856
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 2.71219944954
          sel2 = score <= 10.4189987183
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.6099395752
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.77947998047
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 47.3210754395
          sel3 = llgS <= 14.2293052673
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 17.734582901
          sel2 = chiS <= 11.8465118408
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.8114719391
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4663314819
          sel2 = llgS <= -3.63969087601
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = llgW <= -5.1242351532
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = chiW <= 75.5456542969
          sel2 = chiS <= 10.8906421661
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.4004859924
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7768707275
          sel2 = llgS <= -1.41608667374
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 114.561126709
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.91505622864
          sel2 = llgS <= -2.17533898354
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 18.2651119232
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4667377472
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.726102829
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.8918762207
          sel2 = llgW <= -13.8852863312
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 111.301315308
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.3652648926
          sel2 = llgS <= -3.75050401688
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 23.5625686646
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 69.2014923096
          sel2 = score <= 8.96402740479
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 44.1791610718
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 2.7993183136
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 16.1529216766
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = score <= 23.5550365448
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = score <= 16.8206176758
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgW <= -10.7032260895
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4667377472
          sel2 = chiS <= 9.93839168549
          sel3 = chiW <= 37.7387084961
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = score <= 16.7203807831
          sel2 = chiW <= 34.151473999
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 23.6599769592
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7593231201
          sel2 = score <= 9.14831542969
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 35.7186050415
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgW <= -14.844581604
          sel2 = chiW <= 66.690612793
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 13.0732345581
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.8801422119
          sel2 = score <= 10.7693185806
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.0647220612
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.5741500854
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 100.426437378
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70523118973
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 52.9585266113
          sel3 = llgS <= 12.4256019592
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 16.3652648926
          sel2 = score <= 9.99489784241
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.7931098938
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.75806427
          sel2 = score <= 10.1947011948
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 41.0425338745
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 71.7593231201
          sel2 = score <= 10.0257263184
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 47.3210754395
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 16.8206176758
          sel2 = chiS <= 10.0321826935
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.0723667145
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = chiS <= 10.1260051727
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.818523407
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 73.4189910889
          sel2 = llgS <= -0.727593362331
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.438293457
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.23546981812
          sel2 = chiS <= 10.1315612793
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.6836929321
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.3721885681
          sel2 = llgS <= -2.96500110626
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.7244739532
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6089477539
          sel2 = score <= 9.148229599
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 9.97291660309
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70523118973
          sel2 = score <= 8.45111942291
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 16.4803962708
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.945514679
          sel2 = llgS <= 3.54315519333
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 11.3482933044
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6782512665
          sel2 = score <= 9.27499771118
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.663269043
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4935779572
          sel2 = chiS <= 11.8566045761
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.2558403015
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.0499343872
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = llgS <= 13.5435438156
          sel3 = score <= 9.69693756104
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.65765380859
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 14.5496912003
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = score <= 20.1481781006
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = chiW <= 62.2366027832
          sel2 = llgS <= -1.7924478054
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 112.983093262
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 72.4002990723
          sel2 = score <= 10.2307500839
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.4013633728
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.519323349
          sel2 = chiS <= 10.8906421661
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.7231674194
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = score <= 8.99633598328
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9490699768
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.16190624237
          sel2 = chiS <= 10.1051750183
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 18.5643177032
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.58362960815
          sel2 = score <= 10.4754915237
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 132.548278809
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6108875275
          sel2 = llgS <= -2.59378170967
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.7020053864
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70516014099
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 49.8479728699
          sel3 = llgS <= 12.3758296967
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 3.50358200073
          sel2 = llgS <= -3.19109296799
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 44.1791610718
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 17.495054245
          sel2 = score <= 9.48969078064
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.8020553589
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.68806624413
          sel2 = chiS <= 10.8713264465
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 131.972869873
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8578166962
          sel2 = score <= 9.17582893372
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 24.2407913208
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8206176758
          sel2 = score <= 9.9289226532
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 5.85687351227
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.8881483078
          sel2 = score <= 9.16357421875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.4883346558
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.358417511
          sel2 = llgS <= -2.79083633423
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = llgW <= -3.1143488884
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgS <= 4.52188205719
          sel2 = score <= 10.4189987183
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 16.5995121002
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7800140381
          sel2 = chiS <= 10.8717184067
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 19.4361610413
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 69.2874908447
          sel2 = llgS <= 0.602677583694
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 5.80391263962
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = llgS <= -3.78743720055
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9490699768
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.65709495544
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 12.4566440582
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = chiS <= 60.120639801
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = llgS <= 2.70906686783
          sel2 = chiS <= 12.7661399841
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 18.4286346436
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.0797901154
          sel2 = score <= 20.1049003601
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 17.6806678772
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7768707275
          sel2 = score <= 9.08245563507
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 42.210723877
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 72.4402770996
          sel2 = score <= 9.67998790741
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 47.1310653687
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 17.4667377472
          sel2 = score <= 8.8689956665
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 113.392990112
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6836929321
          sel2 = score <= 9.68029212952
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.661283493
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7593231201
          sel2 = llgS <= -0.755698621273
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 19.6844501495
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8233718872
          sel2 = chiW <= 34.2801704407
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 6.63174772263
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.44084262848
          sel2 = llgW <= -13.5438575745
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.2972145081
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70523118973
          sel2 = score <= 10.0274610519
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 47.7634735107
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 71.75806427
          sel2 = llgW <= -13.6145505905
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          sel3 = score <= 9.63694190979
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgS <= 2.95346593857
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 47.1310653687
          sel3 = score <= 22.7755889893
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 67.5037384033
          sel2 = llgS <= -1.67871069908
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 6.94174194336
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.0434150696
          sel2 = llgS <= 3.5310652256
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 17.4479656219
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 77.3946685791
          sel2 = score <= 9.69707393646
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiW <= 131.901351929
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70516014099
          sel2 = chiS <= 10.9059047699
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.6836929321
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70906686783
          sel2 = score <= 9.59264850616
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.6761817932
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -15.0391368866
          sel2 = llgS <= 3.519323349
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 16.3310375214
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70906686783
          sel2 = llgS <= -2.46436738968
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 18.5608730316
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.0396327972
          sel2 = score <= 9.74484062195
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9424610138
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.936750412
          sel2 = chiS <= 10.8742904663
          sel3 = chiW <= 37.7387084961
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgW <= -14.9040489197
          sel2 = chiW <= 65.5184173584
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 17.5097827911
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8424224854
          sel2 = score <= 9.4867515564
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 8.26961231232
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9444284439
          sel2 = score <= 19.7708892822
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 18.5081958771
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.79832029343
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = score <= 17.7696800232
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = score <= 26.8655700684
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = chiW <= 71.75806427
          sel2 = score <= 9.67998790741
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.4344711304
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9291915894
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = llgS <= 12.4654989243
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = score <= 20.1593418121
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = llgS <= 2.70906686783
          sel2 = score <= 9.16357421875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 16.3310375214
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.8818054199
          sel2 = score <= 10.126581192
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.8322086334
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.71178102493
          sel2 = chiS <= 10.8907613754
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 131.140960693
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.0372848511
          sel2 = llgS <= -2.46436738968
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = chiS <= 11.9521064758
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgS <= 2.68770384789
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 53.5181922913
          sel3 = score <= 23.0303287506
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 2.78729963303
          sel2 = score <= 9.88808917999
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 48.1066818237
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 71.7738037109
          sel2 = score <= 10.5384979248
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 6.59825992584
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9291915894
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = score <= 17.4667377472
          sel3 = score <= 9.61731147766
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.54229259491
          sel2 = llgS <= -3.75035238266
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 18.5643177032
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70491528511
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.3443260193
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.519323349
          sel2 = score <= 9.60803413391
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.3799095154
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.5957069397
          sel2 = chiS <= 10.8866138458
          sel3 = llgS <= -2.71605300903
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = score <= 9.68029212952
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.9779586792
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.2757644653
          sel2 = score <= 9.16357421875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.8450298309
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.31455612183
          sel2 = llgS <= -2.46462464333
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 131.140960693
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.7935962677
          sel2 = score <= 9.16357421875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 15.1303262711
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 72.463508606
          sel2 = chiS <= 9.88936233521
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 44.6230354309
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 71.4365768433
          sel2 = score <= 11.2119207382
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 19.4350280762
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.373632431
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 9.27478408813
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = score <= 25.726102829
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = llgS <= 3.1738858223
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 37.5395736694
          sel3 = score <= 18.5742549896
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 68.5125732422
          sel2 = llgS <= -1.44056665897
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 12.9107637405
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = chiW <= 35.3787879944
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = score <= 9.94221878052
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = llgS <= -2.59590387344
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.871673584
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.5763893127
          sel2 = llgS <= -3.75035238266
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = chiS <= 11.9676933289
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = score <= 16.3716659546
          sel2 = llgS <= -3.16147089005
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = chiS <= 12.8308868408
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgW <= -14.0673980713
          sel2 = llgS <= 3.26720023155
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgW <= 1.02509522438
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.0673980713
          sel2 = chiW <= 62.2740020752
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 17.6723613739
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.7995839119
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 16.1818027496
          sel3 = llgW <= -3.40181159973
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.75806427
          sel2 = llgS <= 0.0935104191303
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 42.2106704712
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 2.65765380859
          sel2 = chiW <= 35.790725708
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.1478672028
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7593231201
          sel2 = score <= 10.4017496109
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.0647220612
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8233718872
          sel2 = score <= 9.67998790741
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 9.88239192963
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.5594177246
          sel2 = llgS <= -2.59590387344
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = score <= 9.19503211975
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = chiW <= 71.8918762207
          sel2 = chiS <= 10.8907260895
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 8.00182533264
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8185081482
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 9.88239192963
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6299533844
          sel2 = score <= 9.19959259033
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.846370697
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.58598446846
          sel2 = score <= 8.94069099426
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 19.6801757812
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8185253143
          sel2 = score <= 10.1373615265
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 106.578598022
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.1141281128
          sel2 = score <= 9.14831542969
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 8.10373592377
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.8918762207
          sel2 = llgS <= -0.794151484966
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 42.1373214722
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 17.097114563
          sel2 = llgS <= -3.75288701057
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = chiS <= 11.6134643555
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = chiW <= 71.7567214966
          sel2 = score <= 10.1469116211
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 7.9680519104
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.983581543
          sel2 = score <= 10.1236553192
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.4339122772
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.4686203003
          sel2 = score <= 8.96402740479
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = chiS <= 14.2343101501
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgW <= -14.0695495605
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = chiW <= 109.578979492
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = score <= 23.211271286
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = llgS <= 3.30449819565
          sel2 = score <= 9.12142944336
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 16.0609588623
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4677352905
          sel2 = chiS <= 9.93922996521
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.6223926544
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9170036316
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = chiW <= 106.274589539
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = score <= 20.6855010986
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = score <= 18.1640815735
          sel2 = chiW <= 34.2492294312
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 8.11863994598
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70523118973
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 16.1782302856
          sel3 = score <= 16.0016689301
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4667377472
          sel2 = chiS <= 12.7661399841
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.7372512817
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70547604561
          sel2 = chiS <= 10.0318088531
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 44.5153579712
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 1.65825664997
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = score <= 20.0598373413
          sel3 = llgW <= -0.651334881783
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.8918762207
          sel2 = score <= 10.0469512939
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 6.59825992584
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 0.959821343422
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 41.0979232788
          sel3 = llgS <= 8.11951065063
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 71.7593231201
          sel2 = llgS <= 0.716506183147
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 45.0309677124
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgW <= -14.9168968201
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = llgS <= 13.0575141907
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = score <= 20.3971576691
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = llgS <= 2.91539430618
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 49.8479728699
          sel3 = llgS <= 14.9965991974
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 3.57897090912
          sel2 = score <= 10.0368642807
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.4677352905
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 77.394821167
          sel2 = chiS <= 10.8863744736
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 51.2910766602
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 3.51923537254
          sel2 = score <= 10.5482730865
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.7352600098
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.5734424591
          sel2 = score <= 9.14894676208
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 7.96187400818
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 18.4658184052
          sel2 = llgS <= -2.66670274734
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9470539093
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 65.4640655518
          sel2 = score <= 10.1465625763
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 41.2418136597
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 16.4920082092
          sel2 = chiS <= 10.0555992126
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.7931098938
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 66.7703857422
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 7.9680519104
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = chiS <= 48.5252876282
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = score <= 17.009557724
          sel2 = llgS <= -3.17125272751
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 24.4177322388
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70523118973
          sel2 = chiW <= 37.7601013184
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 18.5643177032
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9339370728
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = score <= 17.6395759583
          sel3 = score <= 8.67750453949
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.75806427
          sel2 = score <= 9.57650279999
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.3881225586
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 69.4434280396
          sel2 = score <= 10.1654853821
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 41.1761054993
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 2.71583414078
          sel2 = llgS <= -2.22386264801
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.3806667328
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.50358200073
          sel2 = score <= 9.17611885071
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 131.176544189
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.3296356201
          sel2 = chiS <= 10.0050487518
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 9.89177894592
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.24774479866
          sel2 = score <= 9.70509338379
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.4413471222
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6089477539
          sel2 = score <= 9.16347885132
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 6.5605802536
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.88514566422
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 16.3361778259
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.0432872772
          sel2 = llgS <= 2.92018198967
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 17.2914085388
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6089477539
          sel2 = score <= 9.69171714783
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 111.413803101
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.4230079651
          sel2 = chiS <= 10.8864459991
          sel3 = chiW <= 35.048034668
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgS <= 2.10279846191
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 16.1195964813
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = score <= 23.5564880371
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = chiW <= 71.8918762207
          sel2 = llgS <= 0.861723661423
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 5.91751384735
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = llgS <= -3.19109296799
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9490699768
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9663610458
          sel2 = chiW <= 65.214553833
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 16.4309692383
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4664402008
          sel2 = score <= 9.19838523865
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 9.96028327942
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.71378707886
          sel2 = chiS <= 13.0285701752
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 16.1782302856
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.3959941864
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 47.7831420898
          sel3 = llgS <= 9.89199638367
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 17.7336044312
          sel2 = llgS <= -3.1913433075
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.7700176239
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6830615997
          sel2 = score <= 9.27499771118
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 8.11653900146
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = score <= 9.16357421875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 111.301315308
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 70.3531723022
          sel2 = score <= 9.68701553345
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 21.4095916748
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4489860535
          sel2 = chiS <= 9.48193359375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 6.63197946548
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.4520530701
          sel2 = score <= 9.0055103302
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 7.91202640533
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9455184937
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = llgS <= 14.5469112396
          sel3 = score <= 10.006362915
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.9241638184
          sel2 = score <= 10.5394115448
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 19.7403411865
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.87634420395
          sel2 = chiS <= 10.1924591064
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 132.174957275
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.50358200073
          sel2 = llgW <= -13.2649860382
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 16.6273593903
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4716358185
          sel2 = llgS <= -2.7158741951
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9491119385
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 5.15534543991
          sel2 = score <= 9.99423599243
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 16.1300544739
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.8918762207
          sel2 = score <= 10.7491283417
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.1019191742
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.06275987625
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = llgS <= 14.5496912003
          sel3 = chiS <= 24.324420929
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8424224854
          sel2 = score <= 9.69534683228
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 23.7779579163
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.501367569
          sel2 = score <= 9.68628501892
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9491119385
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6089477539
          sel2 = score <= 9.16357421875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 111.399871826
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8233718872
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.0138816833
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4667377472
          sel2 = llgS <= -3.75478863716
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.8146324158
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgW <= -14.9447422028
          p1[self.union_mask([sel1,sel1])] += 1.0
          p2[self.union_mask([sel1,sel1])] += 0.0
          sel2 = llgW <= -0.359420537949
          sel3 = llgS <= 4.18033456802
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7768707275
          sel2 = chiS <= 10.0892791748
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 42.2111320496
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 2.70491528511
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiW <= 132.891433716
          sel3 = score <= 15.8565216064
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 67.9854278564
          sel2 = score <= 10.0087203979
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiW <= 113.242462158
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.4667377472
          sel2 = chiS <= 10.0011615753
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 113.446929932
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.11089134216
          sel2 = chiS <= 12.6439933777
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.4883346558
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.22721886635
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 51.2931747437
          sel3 = score <= 22.1991691589
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 16.7207622528
          sel2 = score <= 9.00492858887
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.726102829
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.3568439484
          sel2 = score <= 8.94086837769
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 100.426879883
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.8807678223
          sel2 = score <= 9.18203353882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 6.58430194855
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.598154068
          sel2 = chiW <= 34.1163520813
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 9.89177894592
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.4953460693
          sel2 = chiS <= 10.126121521
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.0738048553
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8206176758
          sel2 = chiW <= 34.9035949707
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = score <= 9.90094852448
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = score <= 16.5741500854
          sel2 = score <= 9.68029212952
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 7.96187400818
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.70523118973
          sel2 = chiS <= 10.9472551346
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 47.2762069702
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 71.7768707275
          sel2 = score <= 10.3685264587
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 48.1644973755
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 3.23546981812
          sel2 = llgS <= -2.54788923264
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.6007270813
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 67.5707855225
          sel2 = chiS <= 10.0076732635
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.3807277679
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.5547142029
          sel2 = chiW <= 34.2492294312
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 8.37557601929
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6089477539
          sel2 = chiS <= 10.0410432816
          sel3 = score <= 9.69171714783
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = chiW <= 72.4675292969
          sel2 = score <= 10.4189987183
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 19.7109031677
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 4.54565429688
          sel2 = score <= 9.60803413391
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 48.9968490601
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = chiW <= 65.199798584
          sel2 = chiS <= 10.9473934174
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiW <= 111.30329895
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.7935962677
          sel2 = score <= 10.4017496109
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 58.4619827271
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 16.3568878174
          sel2 = score <= 5.42190361023
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgW <= -10.8033905029
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.584022522
          sel2 = llgS <= -3.91831588745
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 9.89729499817
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 66.8005905151
          sel2 = score <= 10.001499176
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = chiS <= 41.5109138489
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = score <= 17.3453292847
          sel2 = score <= 9.69534683228
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.0138816833
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.5594177246
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.8168334961
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 3.44476699829
          sel2 = score <= 9.16347885132
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.6836929321
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.0632972717
          sel2 = llgS <= -3.91831588745
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = chiS <= 12.8977909088
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgS <= 2.88144731522
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 35.7577514648
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = score <= 25.6384410858
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = llgS <= 1.67392492294
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = chiS <= 46.8462600708
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = score <= 25.7911262512
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = score <= 16.8203773499
          sel2 = llgS <= -3.34148836136
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9490699768
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7593231201
          sel2 = llgS <= 0.716506183147
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = llgS <= 7.9680519104
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.6007270813
          sel2 = score <= 8.8689956665
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 26.8055763245
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 18.7210922241
          sel2 = llgS <= -2.26039791107
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = chiS <= 11.5147781372
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = llgS <= 3.89102220535
          sel2 = chiS <= 10.8867673874
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = chiS <= 41.216758728
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = llgS <= 3.28843641281
          sel2 = llgS <= -2.59617137909
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.5594177246
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 16.8206176758
          sel2 = score <= 8.94809341431
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 23.6528167725
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = llgS <= 2.71219944954
          sel2 = score <= 9.17721366882
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 17.7126502991
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.608745575
          sel2 = llgS <= -3.87162733078
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 25.9000911713
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 77.3941345215
          sel2 = score <= 9.16584014893
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 20.3907203674
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = score <= 17.7231674194
          sel2 = llgS <= -2.54788923264
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = llgS <= 10.3812541962
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 68.6917419434
          sel2 = score <= 10.6631288528
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = score <= 19.7453765869
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = chiW <= 71.7800140381
          sel2 = llgW <= -13.5799121857
          p1[self.union_mask([sel1,sel2])] += 1.0
          p2[self.union_mask([sel1,sel2])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = score <= 20.1593418121
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0

          return p1,p2


     def contact_tpfp_rf(self,contact_data):
          p1 = np.zeros(contact_data.shape[0])
          p2 = np.zeros(contact_data.shape[0])

          c1 = contact_data['c1']
          c2 = contact_data['c2']
          c3 = contact_data['c3']
          sl = contact_data['sl']
          sm = contact_data['sm']
          st = contact_data['st']
          cprob = contact_data['cprob']

          sel1 = cprob <= 0.552734375
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.634765625
          sel2 = c2 <= 2.95499992371
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.22499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.24499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c1 <= 3.90500020981
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.61499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = cprob <= 0.373046875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.275390625
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.630859375
          sel2 = c3 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = cprob <= 0.630859375
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.11499977112
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.505859375
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.92500019073
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.41499996185
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.26499986649
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.03499984741
          sel3 = c2 <= 3.42500019073
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.04500007629
          sel3 = cprob <= 0.666015625
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.40500020981
          sel2 = c1 <= 2.93499994278
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.91499996185
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 3.23500013351
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.99499988556
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.42500019073
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.521484375
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.27500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.36328125
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = cprob <= 0.083984375
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 2.95499992371
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.13500022888
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.23500013351
          sel2 = c1 <= 2.86499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.28499984741
          sel2 = cprob <= 0.2109375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.6171875
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = c1 <= 2.88500022888
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.91499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.41499996185
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.654296875
          sel3 = c1 <= 2.74499988556
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.35500001907
          sel3 = cprob <= 0.662109375
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.521484375
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c3 <= 3.05499982834
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.09500026703
          sel3 = c1 <= 3.15500020981
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.04500007629
          sel3 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.630859375
          sel2 = c2 <= 2.91499996185
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.517578125
          sel2 = c1 <= 2.78499984741
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.41499996185
          sel2 = c1 <= 2.90500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.36328125
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.06500005722
          sel3 = c1 <= 3.26499986649
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.21500015259
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.373046875
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.61499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c3 <= 3.15500020981
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = cprob <= 0.099609375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.939453125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.32499980927
          sel3 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.13500022888
          sel3 = cprob <= 0.634765625
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.97499990463
          sel2 = c3 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.630859375
          sel2 = c2 <= 2.95499992371
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.720703125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 3.97499990463
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c3 <= 3.36499977112
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.27500009537
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.490234375
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.349609375
          sel2 = cprob <= 0.091796875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.67500019073
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c3 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.19500017166
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.61499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.30500030518
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.34765625
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.16499996185
          sel3 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.04500007629
          sel3 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.42500019073
          sel2 = c2 <= 3.13500022888
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.365234375
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.10500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.63671875
          sel2 = c1 <= 2.70499992371
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.962890625
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.630859375
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 3.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = cprob <= 0.373046875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.82499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.365234375
          sel2 = sl <= 0.5
          sel3 = c3 <= 3.15500020981
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = st <= 0.5
          sel2 = cprob <= 0.630859375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.251953125
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sl <= 0.5
          sel2 = cprob <= 0.513671875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c3 <= 4.27500009537
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c1 <= 3.91499996185
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.30500030518
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.98500013351
          sel2 = c3 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = c3 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.72499990463
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.13500022888
          sel3 = c3 <= 3.44500017166
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.98500013351
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.25500011444
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = cprob <= 0.197265625
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.91499996185
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.21500015259
          sel3 = cprob <= 0.748046875
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.01500034332
          sel3 = c1 <= 3.25500011444
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.94500017166
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.29500007629
          sel3 = cprob <= 0.720703125
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.337890625
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.19499969482
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c2 <= 2.88500022888
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.14499998093
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.36328125
          sel2 = c2 <= 3.08500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.25500011444
          sel2 = c3 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.634765625
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sl <= 0.5
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c3 <= 4.29500007629
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.748046875
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.91499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.97499990463
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = cprob <= 0.517578125
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.962890625
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.498046875
          sel2 = cprob <= 0.095703125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = sl <= 0.5
          sel2 = cprob <= 0.548828125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c3 <= 4.16499996185
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 3.99499988556
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = c3 <= 3.43499994278
          sel2 = cprob <= 0.265625
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.630859375
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = cprob <= 0.095703125
          sel3 = c2 <= 2.83500003815
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.22499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = cprob <= 0.513671875
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c3 <= 3.42500019073
          sel2 = cprob <= 0.08203125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.513671875
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.958984375
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.36328125
          sel2 = c1 <= 2.61499977112
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 2.97499990463
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = c1 <= 2.84499979019
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.91499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.630859375
          sel2 = c1 <= 2.72499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.15499973297
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.23500013351
          sel2 = c2 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.16499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.21500015259
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.43499994278
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = c1 <= 2.91499996185
          sel2 = c1 <= 2.59499979019
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.744140625
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.08500003815
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.91499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.30500030518
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = cprob <= 0.505859375
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.513671875
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.365234375
          sel2 = c2 <= 2.84499979019
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.61499977112
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.26499986649
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.615234375
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.05500030518
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.26500034332
          sel3 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.953125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 3.97499990463
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c3 <= 3.40500020981
          sel2 = cprob <= 0.162109375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.513671875
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c3 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.06500005722
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = cprob <= 0.361328125
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = cprob <= 0.533203125
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.962890625
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.337890625
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.29500007629
          sel3 = c1 <= 3.26499986649
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.630859375
          sel2 = c3 <= 3.05499982834
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = cprob <= 0.349609375
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.548828125
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.16499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.615234375
          sel3 = c1 <= 2.69500017166
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.22499990463
          sel2 = cprob <= 0.205078125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c3 <= 3.26499986649
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          sel2 = c1 <= 2.875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.62890625
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.27500009537
          sel3 = c1 <= 2.95499992371
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.513671875
          sel2 = cprob <= 0.091796875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.96500015259
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.04500007629
          sel3 = c2 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.16499996185
          sel3 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.91499996185
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.30499982834
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.92500019073
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = cprob <= 0.505859375
          sel2 = c3 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = cprob <= 0.501953125
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c2 <= 2.89499998093
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.748046875
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.568359375
          sel2 = st <= 1.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.06500005722
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.533203125
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.01500034332
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.06500005722
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.361328125
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.21500015259
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c3 <= 3.41499996185
          sel2 = c2 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.42500019073
          sel2 = c1 <= 2.82499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.34765625
          sel2 = c3 <= 3.09499979019
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.21500015259
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.95499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.17500019073
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.28499984741
          sel2 = c1 <= 2.83500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.06500005722
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.634765625
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.27500009537
          sel2 = cprob <= 0.208984375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.625
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.02500009537
          sel3 = c1 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sl <= 0.5
          sel2 = cprob <= 0.337890625
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 4.04500007629
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c3 <= 3.43499994278
          sel2 = cprob <= 0.240234375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.91499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.11499977112
          sel3 = cprob <= 0.962890625
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c3 <= 3.06500005722
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.373046875
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = cprob <= 0.630859375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.91499996185
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.17500019073
          sel3 = cprob <= 0.953125
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.525390625
          sel2 = c3 <= 3.07499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.26500034332
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.513671875
          sel2 = sl <= 0.5
          sel3 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = cprob <= 0.517578125
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.962890625
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.95499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.517578125
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.630859375
          sel2 = c2 <= 3.05499982834
          sel3 = sl <= 0.5
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = cprob <= 0.361328125
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = cprob <= 0.337890625
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.20499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = cprob <= 0.634765625
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = sl <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.513671875
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.15499973297
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          sel2 = cprob <= 0.091796875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.95499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.25500011444
          sel2 = cprob <= 0.197265625
          sel3 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.373046875
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 3.92500019073
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.720703125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.41499996185
          sel3 = c2 <= 3.22499990463
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.513671875
          sel2 = cprob <= 0.099609375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          sel2 = c1 <= 2.82499980927
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.13500022888
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.40500020981
          sel2 = c2 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.517578125
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sl <= 0.5
          sel2 = cprob <= 0.630859375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.91000008583
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.16499996185
          sel3 = cprob <= 0.744140625
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.41499996185
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.24499988556
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = cprob <= 0.380859375
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.755859375
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.06500005722
          sel3 = c1 <= 2.95499992371
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = cprob <= 0.162109375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.26499986649
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.36499977112
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.42500019073
          sel2 = cprob <= 0.060546875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.34375
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.369140625
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.19499969482
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = cprob <= 0.533203125
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.962890625
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.17500019073
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c3 <= 4.25500011444
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.15499973297
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.42500019073
          sel2 = cprob <= 0.1796875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.216796875
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.22499990463
          sel2 = c1 <= 2.85500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.98500013351
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.94500017166
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.25500011444
          sel3 = cprob <= 0.720703125
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.529296875
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.35500001907
          sel3 = c1 <= 3.26499986649
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.42500019073
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.96500015259
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.98500013351
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.35500001907
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.91499996185
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.15500020981
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.10500001907
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = cprob <= 0.517578125
          sel2 = cprob <= 0.091796875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.26499986649
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = sl <= 0.5
          sel2 = c2 <= 3.24499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 4.02500009537
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c3 <= 3.42500019073
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.630859375
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.75500011444
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.04500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.34765625
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.13500022888
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.43499994278
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.26500034332
          sel3 = c1 <= 3.25500011444
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.25500011444
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = cprob <= 0.634765625
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.98500013351
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.15499973297
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.21500015259
          sel2 = c1 <= 2.875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.95499992371
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.541015625
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.25500011444
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.337890625
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.01500034332
          sel3 = c1 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.513671875
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.35500001907
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = sl <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 4.04500007629
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c2 <= 3.22499990463
          sel2 = c2 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = cprob <= 0.091796875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.953125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 2.86499977112
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.630859375
          sel2 = c3 <= 3.24499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = sl <= 0.5
          sel2 = st <= 0.5
          sel3 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c3 <= 3.39499998093
          sel2 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.513671875
          sel2 = c3 <= 3.15500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.25500011444
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = cprob <= 0.173828125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.97499990463
          sel2 = cprob <= 0.091796875
          sel3 = c2 <= 2.83500003815
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c2 <= 3.24499988556
          sel2 = cprob <= 0.197265625
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.634765625
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = cprob <= 0.228515625
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.529296875
          sel2 = cprob <= 0.095703125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.30500030518
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c3 <= 3.42500019073
          sel2 = c1 <= 2.79500007629
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.361328125
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.9609375
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.15499973297
          sel3 = c2 <= 3.41499996185
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.28499984741
          sel2 = cprob <= 0.216796875
          sel3 = c1 <= 2.71500015259
          p1[self.union_mask([sel1,sel2,sel3])] += 0.0
          p2[self.union_mask([sel1,sel2,sel3])] += 1.0
          p1[self.union_mask([sel1,sel2,np.invert(sel3)])] += 0.0
          p2[self.union_mask([sel1,sel2,np.invert(sel3)])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p1[np.invert(sel1)] += 1.0
          p2[np.invert(sel1)] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.27500009537
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.25500011444
          sel3 = c3 <= 3.35500001907
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c1 <= 3.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c2 <= 3.26499986649
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c3 <= 3.27500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.630859375
          sel2 = cprob <= 0.091796875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.03499984741
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.05500030518
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.125
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.09500026703
          sel3 = cprob <= 0.634765625
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.03499984741
          sel3 = cprob <= 0.962890625
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.26499986649
          sel2 = c1 <= 2.875
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.513671875
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.11499977112
          sel3 = c1 <= 3.28499984741
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.541015625
          sel2 = sl <= 0.5
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.02500009537
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.716796875
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.25500011444
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c2 <= 3.23500013351
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.02500009537
          sel3 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = cprob <= 0.630859375
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.27500009537
          sel3 = c3 <= 3.41499996185
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c1 <= 2.95499992371
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = cprob <= 0.34765625
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.23500013351
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 4.08500003815
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = sl <= 0.5
          sel2 = cprob <= 0.517578125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          sel3 = c2 <= 4.04500007629
          p1[self.union_mask([sel1,np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2),np.invert(sel3)])] += 0.0
          p1[np.invert(sel1)] += 0.0
          p2[np.invert(sel1)] += 1.0
          sel1 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c1 <= 3.17500019073
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c3 <= 4.27500009537
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          sel1 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 3.28499984741
          sel3 = cprob <= 0.236328125
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.41499996185
          sel2 = c1 <= 2.79500007629
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c2 <= 3.22499990463
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.634765625
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          sel3 = c3 <= 4.05500030518
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c1 <= 2.99499988556
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.953125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c2 <= 3.98500013351
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c3 <= 3.42500019073
          sel2 = c2 <= 3.13500022888
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c2 <= 3.28499984741
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.03499984741
          sel3 = cprob <= 0.607421875
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.349609375
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = cprob <= 0.953125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          sel3 = c1 <= 3.90500020981
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2),np.invert(sel3)])] += 0.0
          sel1 = c3 <= 3.43499994278
          sel2 = cprob <= 0.162109375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.513671875
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.04500007629
          sel3 = c2 <= 3.32499980927
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.373046875
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c3 <= 4.16499996185
          sel3 = c2 <= 3.36499977112
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.34765625
          sel2 = c1 <= 2.74499988556
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.125
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.97499990463
          sel2 = c2 <= 2.83500003815
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = cprob <= 0.962890625
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.337890625
          sel2 = c1 <= 2.69500017166
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.29500007629
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = cprob <= 0.337890625
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = c2 <= 4.03499984741
          sel3 = c1 <= 3.25500011444
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.41499996185
          sel2 = c2 <= 3.125
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c3 <= 4.30500030518
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = sl <= 0.5
          sel2 = cprob <= 0.630859375
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.34499979019
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = cprob <= 0.36328125
          sel2 = c2 <= 2.91499996185
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c2 <= 4.06500005722
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 2.85500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = st <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = st <= 0.5
          sel2 = c1 <= 2.97499990463
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = c2 <= 3.22499990463
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.42500019073
          p1[self.union_mask([sel1,sel1])] += 0.0
          p2[self.union_mask([sel1,sel1])] += 1.0
          sel2 = sl <= 0.5
          sel3 = cprob <= 0.513671875
          p1[self.union_mask([np.invert(sel1),sel2,sel3])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2,sel3])] += 1.0
          p1[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2,np.invert(sel3)])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = st <= 0.5
          sel2 = c3 <= 3.40500020981
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          sel2 = cprob <= 0.251953125
          p1[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          sel1 = c3 <= 3.40500020981
          sel2 = c2 <= 3.10500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = sm <= 0.5
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          sel1 = c1 <= 2.99499988556
          sel2 = c2 <= 2.85500001907
          p1[self.union_mask([sel1,sel2])] += 0.0
          p2[self.union_mask([sel1,sel2])] += 1.0
          p1[self.union_mask([sel1,np.invert(sel2)])] += 0.0
          p2[self.union_mask([sel1,np.invert(sel2)])] += 1.0
          sel2 = c1 <= 3.26499986649
          p1[self.union_mask([np.invert(sel1),sel2])] += 1.0
          p2[self.union_mask([np.invert(sel1),sel2])] += 0.0
          p1[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 1.0
          p2[self.union_mask([np.invert(sel1),np.invert(sel2)])] += 0.0


          return p1,p2
          

"""
Original trees output by sci-kit learn module
parsed to individual functions

def dtree_0(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c3 <= 3.46500015259:
          if st <= 0.5:
               if c3 <= 3.20499992371:
                    return (95.0,873.926460001)
               else:
                    return (302.0,361.15002)
          else:
               return (258.0,2446.47414)
     else:
          if sl <= 0.5:
               if c2 <= 3.24499988556:
                    return (158.0,348.22656)
               else:
                    return (5362.0,551.754720001)
          else:
               return (103.0,136.08936)
def dtree_1(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          if c2 <= 3.22499990463:
               return (301.0,1401.05592)
          else:
               if c1 <= 3.03499984741:
                    return (289.0,313.26966)
               else:
                    return (5137.0,233.41032)
     else:
          if wt <= 1.5:
               if sl <= 0.5:
                    return (417.0,1362.09942)
               else:
                    return (146.0,1390.75002)
          else:
               return (111.0,16.79238)
def dtree_10(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c2 <= 3.125:
               if c3 <= 2.53499984741:
                    return (37.0,38.87136)
               else:
                    return (367.0,3596.82444001)
          else:
               if c1 <= 2.83500003815:
                    return (198.0,482.749740001)
               else:
                    return (198.0,206.91594)
     else:
          if c2 <= 3.24499988556:
               return (169.0,104.70636)
          else:
               return (5397.0,287.37918)
def dtree_11(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c3 <= 3.39499998093:
          if c2 <= 3.06500005722:
               return (232.0,3100.29588)
          else:
               return (212.0,362.52216)
     else:
          if c2 <= 3.27500009537:
               if c2 <= 3.05499982834:
                    return (69.0,389.40462)
               else:
                    return (175.0,211.44222)
          else:
               if c1 <= 2.99499988556:
                    return (272.0,392.566680001)
               else:
                    return (5312.0,261.40158)
def dtree_12(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sm <= 0.5:
          if c1 <= 3.00500011444:
               if c2 <= 3.125:
                    return (162.0,1653.8049)
               else:
                    return (289.0,462.937860001)
          else:
               return (5317.0,313.83)
     else:
          if wt <= 1.5:
               if c2 <= 3.10500001907:
                    return (185.0,1963.17792)
               else:
                    return (317.0,308.8602)
          else:
               return (31.0,14.96484)
def dtree_13(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c3 <= 3.46500015259:
          if st <= 0.5:
               if c2 <= 3.125:
                    return (141.0,1079.05644)
               else:
                    if c1 <= 2.98500013351:
                         return (60.0,107.3061)
                    else:
                         return (212.0,49.62672)
          else:
               return (267.0,2445.49404)
     else:
          if c2 <= 3.21500015259:
               return (139.0,397.039500001)
          else:
               return (5382.0,639.250920001)
def dtree_14(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.97499990463:
          if c2 <= 3.17500019073:
               if c1 <= 1.90499997139:
                    return (87.0,51.58692)
               else:
                    return (303.0,3667.66686001)
          else:
               if c1 <= 2.91499996185:
                    return (234.0,501.935940001)
               else:
                    return (92.0,65.13804)
     else:
          if c1 <= 3.17500019073:
               return (490.0,252.61038)
          else:
               return (5024.0,178.77816)
def dtree_15(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          return (5791.0,1946.19942)
     else:
          if sl <= 3.5:
               if c2 <= 3.19500017166:
                    return (258.0,2464.57134)
               else:
                    if wt <= 1.5:
                         if c2 <= 3.52500009537:
                              return (151.0,218.02374)
                         else:
                              return (148.0,63.23526)
                    else:
                         return (47.0,0.4554)
          else:
               return (40.0,24.82524)
def dtree_16(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c1 <= 1.97500002384:
               return (122.0,79.18218)
          else:
               if c3 <= 3.36499977112:
                    return (283.0,3230.46306001)
               else:
                    if c1 <= 2.91499996185:
                         return (281.0,905.909400001)
                    else:
                         return (114.0,110.78892)
     else:
          if c2 <= 3.33500003815:
               return (360.0,143.49456)
          else:
               return (5363.0,247.29804)
def dtree_17(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sm <= 0.5:
          if c3 <= 3.41499996185:
               if c1 <= 2.97499990463:
                    return (177.0,1485.40986)
               else:
                    return (130.0,68.94162)
          else:
               if c3 <= 3.52500009537:
                    if c1 <= 2.99499988556:
                         return (63.0,169.20684)
                    else:
                         return (271.0,31.40874)
               else:
                    return (5100.0,677.308500001)
     else:
          return (517.0,2285.3853)
def dtree_18(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sm <= 0.5:
          if c1 <= 2.99499988556:
               if c1 <= 2.83500003815:
                    if c1 <= 2.09499979019:
                         return (77.0,50.88996)
                    else:
                         return (247.0,1732.02084)
               else:
                    return (206.0,323.50824)
          else:
               if c1 <= 3.17500019073:
                    return (397.0,160.29486)
               else:
                    return (4894.0,167.38128)
     else:
          return (474.0,2283.49242)
def dtree_19(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.21500015259:
          if c3 <= 3.20499992371:
               if c1 <= 1.99500000477:
                    return (61.0,66.5082)
               else:
                    return (175.0,2746.92726)
          else:
               return (335.0,1085.81616)
     else:
          if c1 <= 3.02500009537:
               if st <= 0.5:
                    return (295.0,315.07542)
               else:
                    return (84.0,228.80484)
          else:
               return (5391.0,274.36464)
def dtree_2(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          if c1 <= 2.99499988556:
               if c3 <= 3.35500001907:
                    return (133.0,1038.63672)
               else:
                    if c2 <= 3.42500019073:
                         return (144.0,420.466860001)
                    else:
                         return (182.0,169.05042)
          else:
               return (5210.0,319.89276)
     else:
          if c2 <= 3.34499979019:
               return (296.0,2609.9766)
          else:
               return (281.0,159.66126)
def dtree_20(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sm <= 0.5:
          if sl <= 0.5:
               if c1 <= 2.99499988556:
                    if c3 <= 3.35500001907:
                         return (136.0,1035.77364)
                    else:
                         return (332.0,587.610540001)
               else:
                    return (5125.0,321.94602)
          else:
               return (125.0,487.737360001)
     else:
          if c1 <= 3.02500009537:
               return (274.0,2236.32684)
          else:
               return (226.0,48.34566)
def dtree_21(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if st <= 0.5:
               if c3 <= 3.26499986649:
                    return (100.0,930.746520001)
               else:
                    return (307.0,695.275020001)
          else:
               if c1 <= 1.875:
                    return (55.0,34.4025)
               else:
                    return (291.0,2667.2283)
     else:
          if c3 <= 3.36499977112:
               return (71.0,68.31792)
          else:
               return (5423.0,321.71238)
def dtree_22(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.21500015259:
          return (610.0,3898.74672001)
     else:
          if c2 <= 3.41499996185:
               if c3 <= 3.39499998093:
                    return (93.0,107.69616)
               else:
                    return (527.0,225.51606)
          else:
               if st <= 0.5:
                    return (4945.0,368.46612)
               else:
                    if c1 <= 3.04500007629:
                         return (57.0,101.3166)
                    else:
                         return (227.0,15.52122)
def dtree_23(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          if c2 <= 3.22499990463:
               if c2 <= 3.05499982834:
                    return (157.0,1174.93398)
               else:
                    return (170.0,224.21916)
          else:
               return (5465.0,545.628600001)
     else:
          if c2 <= 3.39499998093:
               if c3 <= 3.42500019073:
                    return (233.0,2379.21156)
               else:
                    return (112.0,264.627)
          else:
               return (268.0,128.7495)
def dtree_24(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sm <= 0.5:
          if c1 <= 2.99499988556:
               return (470.0,2107.76742)
          else:
               return (5250.0,326.81682)
     else:
          if c1 <= 3.04500007629:
               if c1 <= 1.875:
                    return (61.0,30.95334)
               else:
                    if c2 <= 3.07499980927:
                         return (124.0,1902.80772)
                    else:
                         return (104.0,309.03444)
          else:
               return (206.0,40.36626)
def dtree_25(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c2 <= 3.17500019073:
               if c1 <= 1.98500001431:
                    return (77.0,75.54294)
               else:
                    return (317.0,3661.88526001)
          else:
               if c2 <= 3.47499990463:
                    return (172.0,374.85954)
               else:
                    return (169.0,212.68962)
     else:
          if c3 <= 3.34499979019:
               return (67.0,63.31446)
          else:
               return (5524.0,329.2344)
def dtree_26(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c1 <= 2.18499994278:
               return (163.0,209.77704)
          else:
               if c1 <= 2.91499996185:
                    if c3 <= 3.39499998093:
                         return (183.0,3107.13282)
                    else:
                         return (263.0,810.576360001)
               else:
                    if c2 <= 3.21500015259:
                         return (50.0,123.30054)
                    else:
                         return (104.0,76.51512)
     else:
          return (5530.0,390.28968)
def dtree_27(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.22499990463:
          if st <= 0.5:
               return (328.0,1400.24808)
          else:
               if c1 <= 1.95500004292:
                    return (59.0,49.40496)
               else:
                    return (218.0,2469.30354)
     else:
          if c1 <= 2.99499988556:
               return (316.0,504.108000001)
          else:
               if c1 <= 3.17500019073:
                    return (398.0,120.43746)
               else:
                    return (4968.0,174.1014)
def dtree_28(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c1 <= 2.09499979019:
               return (137.0,137.54862)
          else:
               if c3 <= 3.36499977112:
                    return (257.0,3179.27610001)
               else:
                    return (397.0,1008.98028)
     else:
          if c3 <= 3.35500001907:
               return (79.0,66.27852)
          else:
               if c1 <= 3.17500019073:
                    return (470.0,153.50346)
               else:
                    return (5001.0,171.90954)
def dtree_29(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          if c1 <= 2.97499990463:
               if c2 <= 3.18499994278:
                    return (181.0,1291.16196)
               else:
                    return (238.0,306.24858)
          else:
               if c2 <= 3.24499988556:
                    return (147.0,85.041)
               else:
                    return (5016.0,261.7065)
     else:
          if c1 <= 3.04500007629:
               return (323.0,2729.8755)
          else:
               return (279.0,43.77384)
def dtree_3(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          if c2 <= 3.22499990463:
               return (290.0,1398.57498)
          else:
               if c1 <= 2.99499988556:
                    return (233.0,289.51164)
               else:
                    return (5152.0,258.03558)
     else:
          if c3 <= 3.47499990463:
               if c1 <= 2.19500017166:
                    return (116.0,152.9649)
               else:
                    return (158.0,2306.601)
          else:
               return (356.0,311.8797)
def dtree_30(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sl <= 0.5:
          if st <= 0.5:
               return (5557.0,1944.6273)
          else:
               if c3 <= 3.42500019073:
                    if c2 <= 3.14499998093:
                         return (97.0,1078.24068)
                    else:
                         return (40.0,67.77144)
               else:
                    return (274.0,223.06284)
     else:
          if wt <= 1.5:
               return (134.0,1389.3858)
          else:
               return (80.0,14.72328)
def dtree_31(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          if c2 <= 3.22499990463:
               if c1 <= 2.90500020981:
                    return (165.0,1253.14992)
               else:
                    return (153.0,144.61722)
          else:
               if c1 <= 3.08500003815:
                    return (373.0,341.15796)
               else:
                    return (4961.0,205.19532)
     else:
          if c1 <= 3.02500009537:
               return (353.0,2720.99718)
          else:
               return (264.0,52.52148)
def dtree_32(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c1 <= 1.98500001431:
               return (103.0,82.30464)
          else:
               if sl <= 0.5:
                    if c2 <= 3.17500019073:
                         return (264.0,2436.02964)
                    else:
                         return (306.0,459.316440001)
               else:
                    return (74.0,1348.35822)
     else:
          if c1 <= 3.17500019073:
               return (503.0,211.9986)
          else:
               return (5079.0,179.51274)
def dtree_33(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sm <= 0.5:
          if c3 <= 3.38500022888:
               return (252.0,1500.14106)
          else:
               if c2 <= 3.24499988556:
                    return (178.0,393.726960001)
               else:
                    return (5358.0,540.310320001)
     else:
          if c2 <= 3.22499990463:
               if c1 <= 1.99500000477:
                    return (52.0,53.86392)
               else:
                    return (162.0,2036.3112)
          else:
               return (260.0,193.29948)
def dtree_34(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.22499990463:
          if wt <= 0.5:
               return (573.0,3858.80814001)
          else:
               return (43.0,58.26744)
     else:
          if c3 <= 3.46500015259:
               if c1 <= 2.95499992371:
                    return (53.0,123.55794)
               else:
                    return (169.0,44.7183)
          else:
               if c2 <= 3.47499990463:
                    return (752.0,228.24648)
               else:
                    return (4759.0,403.882380001)
def dtree_35(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.22499990463:
          if c1 <= 2.97499990463:
               if c3 <= 2.47499990463:
                    return (36.0,25.1163)
               else:
                    return (443.0,3772.50390001)
          else:
               return (160.0,119.5227)
     else:
          if st <= 0.5:
               if c2 <= 3.41499996185:
                    return (499.0,178.51482)
               else:
                    return (4760.0,367.96122)
          else:
               return (315.0,254.13102)
def dtree_36(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.22499990463:
          if c1 <= 2.99499988556:
               if c1 <= 2.18499994278:
                    return (136.0,193.75092)
               else:
                    return (302.0,3629.66472001)
          else:
               return (141.0,95.50134)
     else:
          if st <= 0.5:
               if c2 <= 3.38500022888:
                    return (403.0,152.88768)
               else:
                    return (5024.0,393.372540001)
          else:
               return (312.0,252.36486)
def dtree_37(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sl <= 0.5:
          if st <= 0.5:
               if c2 <= 3.22499990463:
                    return (331.0,1398.969)
               else:
                    if c1 <= 2.99499988556:
                         return (234.0,290.30562)
                    else:
                         return (5046.0,256.32288)
          else:
               if c1 <= 3.02500009537:
                    return (201.0,1322.06976)
               else:
                    return (187.0,44.3322)
     else:
          return (212.0,1405.75446)
def dtree_38(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c3 <= 3.46500015259:
          if c2 <= 3.06500005722:
               return (332.0,3209.06322001)
          else:
               if st <= 0.5:
                    return (320.0,215.11314)
               else:
                    return (99.0,257.73462)
     else:
          if c1 <= 2.99499988556:
               return (350.0,746.883720001)
          else:
               if c1 <= 3.17500019073:
                    return (340.0,126.07452)
               else:
                    return (4940.0,162.5481)
def dtree_39(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sl <= 0.5:
          if c2 <= 3.19500017166:
               if c1 <= 2.97499990463:
                    return (313.0,2474.92674)
               else:
                    return (117.0,95.52906)
          else:
               if c1 <= 3.02500009537:
                    return (383.0,464.684220001)
               else:
                    return (5274.0,277.12278)
     else:
          if sl <= 1.5:
               return (100.0,1151.85114)
          else:
               return (135.0,253.4202)
def dtree_4(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c2 <= 3.125:
               if wt <= 0.5:
                    return (339.0,3581.66754001)
               else:
                    return (43.0,55.78452)
          else:
               return (398.0,689.061780001)
     else:
          if c2 <= 3.24499988556:
               return (166.0,104.68854)
          else:
               if c1 <= 3.17500019073:
                    return (367.0,115.0677)
               else:
                    return (4942.0,171.39672)
def dtree_40(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if wt <= 0.5:
               if st <= 0.5:
                    if c1 <= 2.91499996185:
                         return (310.0,1503.45558)
                    else:
                         return (136.0,120.87702)
               else:
                    return (263.0,2642.8248)
          else:
               return (46.0,57.7269)
     else:
          if c1 <= 3.15500020981:
               return (385.0,198.67518)
          else:
               return (5211.0,193.91724)
def dtree_41(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.22499990463:
          return (554.0,3917.95866001)
     else:
          if st <= 0.5:
               if c2 <= 3.41499996185:
                    if c1 <= 2.95499992371:
                         return (49.0,104.86674)
                    else:
                         return (420.0,73.28574)
               else:
                    return (4869.0,369.28188)
          else:
               if sl <= 1.5:
                    return (259.0,245.74374)
               else:
                    return (82.0,6.5736)
def dtree_42(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.24499988556:
          if c1 <= 2.99499988556:
               if wt <= 0.5:
                    return (428.0,3795.14322001)
               else:
                    return (48.0,56.52108)
          else:
               return (172.0,104.67666)
     else:
          if st <= 0.5:
               if c3 <= 3.46500015259:
                    return (169.0,72.50166)
               else:
                    return (5235.0,452.572560001)
          else:
               return (299.0,236.06154)
def dtree_43(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          if c2 <= 3.18499994278:
               return (275.0,1353.35178)
          else:
               if c1 <= 2.99499988556:
                    return (259.0,321.16788)
               else:
                    return (5082.0,269.6661)
     else:
          if wt <= 1.5:
               if c1 <= 3.02500009537:
                    return (300.0,2703.46824)
               else:
                    return (223.0,52.51356)
          else:
               return (87.0,17.55666)
def dtree_44(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c3 <= 3.46500015259:
          if c1 <= 3.04500007629:
               if c1 <= 1.97500002384:
                    return (93.0,68.55552)
               else:
                    if c1 <= 2.88500022888:
                         return (272.0,3333.99924001)
                    else:
                         return (112.0,215.1171)
          else:
               return (218.0,63.79164)
     else:
          if st <= 0.5:
               return (5317.0,709.887420001)
          else:
               return (345.0,326.11392)
def dtree_45(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.22499990463:
          if c2 <= 3.07499980927:
               if wt <= 1.5:
                    return (291.0,3515.93154001)
               else:
                    return (22.0,17.02404)
          else:
               if c3 <= 3.35500001907:
                    return (82.0,218.5029)
               else:
                    return (134.0,168.00696)
     else:
          if c1 <= 2.99499988556:
               return (338.0,502.773480001)
          else:
               return (5411.0,295.38234)
def dtree_46(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sm <= 0.5:
          if c1 <= 2.95499992371:
               if c3 <= 3.36499977112:
                    return (126.0,1397.28798)
               else:
                    return (287.0,648.893520001)
          else:
               return (5356.0,386.35542)
     else:
          if c1 <= 3.04500007629:
               if c2 <= 3.10500001907:
                    return (170.0,1972.8621)
               else:
                    return (115.0,272.2401)
          else:
               return (185.0,40.05936)
def dtree_47(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c2 <= 3.125:
               if c1 <= 2.19500017166:
                    return (155.0,197.35848)
               else:
                    if c1 <= 2.86499977112:
                         return (211.0,3213.71226001)
                    else:
                         return (70.0,222.65298)
          else:
               return (379.0,691.776360001)
     else:
          if c1 <= 3.17500019073:
               return (474.0,212.50746)
          else:
               return (4949.0,179.69292)
def dtree_48(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sl <= 0.5:
          if c1 <= 2.95499992371:
               if c3 <= 3.26499986649:
                    if c1 <= 2.17500019073:
                         return (57.0,57.54078)
                    else:
                         return (135.0,1824.80562)
               else:
                    return (374.0,969.859440001)
          else:
               if st <= 0.5:
                    return (5341.0,376.73064)
               else:
                    return (224.0,82.2987)
     else:
          return (217.0,1406.24748)
def dtree_49(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c2 <= 3.22499990463:
          if c2 <= 3.10500001907:
               if wt <= 0.5:
                    if c1 <= 1.98500001431:
                         return (57.0,48.17538)
                    else:
                         return (332.0,3520.67958001)
               else:
                    return (33.0,56.46762)
          else:
               return (207.0,290.02842)
     else:
          if c1 <= 2.99499988556:
               return (331.0,505.523700001)
          else:
               return (5318.0,296.74656)
def dtree_5(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c1 <= 2.19500017166:
               return (176.0,218.51082)
          else:
               if st <= 0.5:
                    if c2 <= 3.22499990463:
                         return (157.0,1285.6734)
                    else:
                         return (228.0,277.84152)
               else:
                    return (188.0,2543.44068)
     else:
          if c2 <= 3.33500003815:
               return (335.0,143.88066)
          else:
               return (5267.0,248.12964)
def dtree_6(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c3 <= 3.41499996185:
          if c2 <= 3.18499994278:
               return (361.0,3360.99852001)
          else:
               if st <= 0.5:
                    return (154.0,79.21782)
               else:
                    if c1 <= 2.99499988556:
                         return (9.0,75.89934)
                    else:
                         return (19.0,9.67626)
     else:
          if c1 <= 2.99499988556:
               return (409.0,885.380760001)
          else:
               return (5433.0,306.2367)
def dtree_7(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if sl <= 0.5:
          if c1 <= 2.97499990463:
               return (605.0,2897.13798)
          else:
               if st <= 0.5:
                    return (5288.0,346.62276)
               else:
                    return (240.0,68.93568)
     else:
          if wt <= 0.5:
               return (96.0,1357.16526)
          else:
               if c1 <= 2.86499977112:
                    return (34.0,46.36368)
               else:
                    return (61.0,1.30482)
def dtree_8(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if st <= 0.5:
          if c2 <= 3.22499990463:
               if c2 <= 3.06500005722:
                    return (150.0,1191.41352)
               else:
                    return (147.0,207.3555)
          else:
               if c1 <= 3.00500011444:
                    return (267.0,295.09326)
               else:
                    return (5223.0,251.19864)
     else:
          if c1 <= 3.01499986649:
               return (359.0,2713.98402)
          else:
               return (313.0,58.21794)
def dtree_9(params):
     'wt','sl','sm','st','c1','c2','c3' = params
     if c1 <= 2.99499988556:
          if c1 <= 1.98500001431:
               return (101.0,82.18386)
          else:
               if c2 <= 3.03499984741:
                    return (220.0,3305.83770001)
               else:
                    if c2 <= 3.35500001907:
                         return (194.0,608.061960001)
                    else:
                         return (250.0,330.83028)
     else:
          if c1 <= 3.17500019073:
               return (460.0,211.79466)
          else:
               return (4989.0,179.03952)
###########
code to convet the above to vectorized mask_array form
--> very hacked together, sorry.

import os,sys
import numpy as np

f = open(sys.argv[1],'r')
func = f.readlines()
indent = []
if_pos = 0
ret_pos = 0
el_pos = 0
selno = 0
mask_list = []
mask_strlist = []
for findex,line in enumerate(func):
    words = line.split(' ')
    for windex,word in enumerate(words):
        words[windex] = word.replace('\n','')
    for index,word in enumerate(words):
        if word[0:2] == "if":
            indent.append(index)
            if_pos = index
            #print "INDEX",indent.index(if_pos), findex,index,indent

            #print "IF",indent.index(if_pos)
            #print indent
            if_ind = indent.index(if_pos)
            selno = if_ind + 1
            if selno == 1:
                mask_list = [1,]
                mask_strlist = ['sel1',]
                indent = [if_pos,]
            else:
                new_mask_list = []
                new_mask_strlist = []
                new_indent = []
                for i in range(if_ind):
                    new_mask_list.append(mask_list[i])
                    new_mask_strlist.append(mask_strlist[i])
                mask_list = new_mask_list
                mask_strlist = new_mask_strlist
                mask_list.append(selno)
            assert indent.index(if_pos) == len(indent)-1
            mask_strlist.append("sel%s" % str(selno))
            print "sel"+str(selno)+" = "+words[index+1]+" "+words[index+2]+" "+words[index+3].replace(':',"")
        elif word[0:3] == "ret":
            ret_pos = index
            #if just_ret:
            #    mask_strlist[-1] = "np.invert(%s)" % mask_list[-1]
            union_list = ','.join(mask_strlist)
            vals = words[index+1].split(',')
            val1 = vals[0].replace('(','')
            val2 = vals[1].replace(')','')
            if len(mask_strlist) > 1:
                
                print "p1[self.union_mask([%s])] += %s" % (union_list,val1)
                print "p2[self.union_mask([%s])] += %s" % (union_list,val2)
            else:
                print "p1[%s] += %s" % (union_list,val1)
                print "p2[%s] += %s" % (union_list,val2)

        elif word[0:4] == "else":
            #print mask_list
            el_pos = index
            ind_pos = indent.index(el_pos)
            #print "ELSE",ind_pos
            #selno = ind_pos+1
            new_mask_list = []
            new_mask_strlist = []
            new_indent = []
            for i in range(ind_pos + 1):
                new_mask_list.append(mask_list[i])
                new_mask_strlist.append(mask_strlist[i])
                new_indent.append(indent[i])
            mask_list = new_mask_list
            mask_strlist = new_mask_strlist
            indent = new_indent

            mask_strlist[-1] = "np.invert(%s)" % mask_strlist[-1]

        

        
"""
