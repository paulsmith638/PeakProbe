from __future__ import division
import sys,os,math,copy
import numpy as np
#cctbx imports
import iotbx.pdb
from mmtbx import monomer_library
from libtbx import group_args
from cctbx import miller
from scitbx.array_family import flex
from scitbx import matrix
from iotbx import reflection_file_utils
from mmtbx.command_line import geometry_minimization
from mmtbx import monomer_library
import mmtbx.refinement.geometry_minimization
from cctbx import maptbx
import mmtbx.ncs
import mmtbx.refinement.real_space.individual_sites as rsr
import mmtbx.refinement.real_space.rigid_body as rigid
import iotbx.ncs
import mmtbx.maps.correlation
from iotbx.ccp4_map import write_ccp4_map
#PProbe imports
from PProbe_peak import PeakObj
from PProbe_struct import StructData
from  PProbe_ref import RSRefinements
from PProbe_util import Util
from PProbe_cctbx import CctbxHelpers
pptbx = CctbxHelpers()
class RealSpace:
    def __init__(self,peak_object,ref_object,features_dict,ressig=False):
        """
        Class for realspace refinement operations (see PProbe_ref for cctbx refinement routines)
        Various RSR operations are done with results (correlation coefficients, input and output structures, etc.,
        are stored in standard python dictionary.
        peak_object = contains coordinates and maps
        ref_ofject = methods for working with maps 
        """
        self.features = features_dict
        #unbound methods for rsr and cc
        self.rsrefine = ref_object.refine_rsr
        self.calc_rscc = ref_object.calculate_cc
        #pass struct data ref along
        self.struct_data = peak_object.struct_data

        self.resid = peak_object.resid
        self.peak_coord = peak_object.coord
        #initial input data cc, not actually used
        self.features['so4_in_hier'] = peak_object.so4_hier
        self.features['wat_in_hier'] = peak_object.wat_hier
        self.features['so4_cc_fofc_in']=self.rscc(peak_object.so4_hier,peak_object.so4_xrs,"resname SO4",ref_object.fofc_map_data)
        self.features['so4_cc_2fofc_in']=self.rscc(peak_object.so4_hier,peak_object.so4_xrs,"resname SO4",ref_object.twofofc_map_data)
        self.features['wat_cc_fofc_in']=self.rscc(peak_object.wat_hier,peak_object.wat_xrs,"resname HOH",ref_object.fofc_map_data) 
        self.features['wat_cc_2fofc_in']=self.rscc(peak_object.wat_hier,peak_object.wat_xrs,"resname HOH",ref_object.twofofc_map_data)
        #harmonic positional restraints for refinements
        #inherited from struct_data class as they are the same for all peaks and are very slow to construct
        self.tso4_restr = self.struct_data.so4_restraints_01sig#tight restraints
        self.twat_restr = self.struct_data.wat_restraints_01sig#tight restraints
        self.lso4_restr = self.struct_data.so4_restraints_1sig#loose restraints
        self.lwat_restr = self.struct_data.wat_restraints_1sig#loose restraints
        #for adjusting tight restraints on the fly, time costly, avoid if possible
        if ressig > 0: 
            self.lso4_restr = self.struct_data.make_so4_restraints(ressig)
            self.lwat_restr = self.struct_data.make_wat_restraints(ressig)

        #refinement setup
        self.check_local(peak_object,ref_object.twofofc_map_data)


    def check_local(self,peak_object,input_map):
        """
        Check for 2fofc peak close by (like a metal ion) by distance from peak to density maximum
        metal ions typically show a peak 2.5-3.5A away from the candidate peak
        otherwise, max density is peak itself, or is something far away
        if peak is close enough to interfere with 2fofc rsr refinement, use shaped map
        that has a sharp falloff at 2A from peak center
        """
        pmax_dist = peak_object.find_peaks(input_map)
        if pmax_dist > 2.3 and pmax_dist < 4.5:
            self.target_2fofc = peak_object.shaped_map_2fofc
        else:
            self.target_2fofc = peak_object.local_map_2fofc


    def refinement(self,peak,peak_object,ref_object,write_pdb=False,outstr=''):
        spdb_in = peak_object.so4_pdb
        shier_in = peak_object.so4_hier
        sxrs_in = peak_object.so4_xrs
        wpdb_in = peak_object.wat_pdb
        whier_in = peak_object.wat_hier
        wxrs_in = peak_object.wat_xrs

        #First, refine sulfate against fofc with loose restraints
        self.features['so4_fofc_ref_xrs'],self.features['so4_fofc_ref_hier'],dmove = self.prsr(spdb_in,shier_in,sxrs_in,peak_object.local_map_fofc,self.lso4_restr)

        #if we moved too far, try again with tighter restraints
        if dmove > 1.8:
            self.features['so4_fofc_ref_xrs'],self.features['so4_fofc_ref_hier'],dmove = self.prsr(spdb_in,shier_in,sxrs_in,peak_object.local_map_fofc,self.lso4_restr)
        #if still no good, try shaped map that truncates density away from the peak

        if dmove > 1.8:
            self.features['so4_fofc_ref_xrs'],self.features['so4_fofc_ref_hier'],dmove = self.prsr(spdb_in,shier_in,sxrs_in,peak_object.shaped_map_fofc,self.tso4_restr)

        #all cc's are against original maps
        self.features['so4_cc_fofc_out']=self.rscc(self.features['so4_fofc_ref_hier'],self.features['so4_fofc_ref_xrs'],"resname SO4",ref_object.fofc_map_data)
        self.features['so4_fofc_coord_out'] = self.features['so4_fofc_ref_hier'].atoms()[0].xyz

        if write_pdb:
            ref_object.write_pdb_file(self.features['so4_in_hier'],peak['db_id']+"_s_refin.pdb")
            ref_object.write_pdb_file(self.features['so4_fofc_ref_hier'],peak['db_id']+"_fofc_"+str(outstr)+"_s_refout.pdb")

        #Now refine against 2FOFC using similar procedure
        #will use shaped map if strong 2fofc peak closeby, otherwise original map, try loose restraints first

        self.features['so4_2fofc_ref_xrs'],self.features['so4_2fofc_ref_hier'],dmove = self.prsr(spdb_in,shier_in,sxrs_in,self.target_2fofc,self.lso4_restr)
        if dmove > 1.8:
            self.features['so4_2fofc_ref_xrs'],self.features['so4_2fofc_ref_hier'],dmove = self.prsr(spdb_in,shier_in,sxrs_in,self.target_2fofc,self.tso4_restr)

        self.features['so4_cc_2fofc_out']=self.rscc(self.features['so4_2fofc_ref_hier'],self.features['so4_2fofc_ref_xrs'],"resname SO4",ref_object.twofofc_map_data)
        self.features['so4_2fofc_shift'] = self.xform_to_original(shier_in,self.features['so4_2fofc_ref_hier'],self.peak_coord)        
        pptbx.renumber_residue(self.features['so4_2fofc_shift'],self.resid)
        if write_pdb:
            ref_object.write_pdb_file(self.features['so4_2fofc_ref_hier'],peak['db_id']+"_2fofc_"+str(outstr)+"_s_refout.pdb")
            ref_object.write_pdb_file(self.features['so4_2fofc_shift'],peak['db_id']+"_2fofc_"+str(outstr)+"_origS_refout.pdb")            


        #repeat same procedure for water
        self.features['wat_fofc_ref_xrs'],self.features['wat_fofc_ref_hier'],dmove = self.prsr(wpdb_in,whier_in,wxrs_in,peak_object.local_map_fofc,self.lwat_restr)
        if dmove > 2.5:
            self.features['wat_fofc_ref_xrs'],self.features['wat_fofc_ref_hier'],dmove = self.prsr(wpdb_in,whier_in,wxrs_in,peak_object.local_map_fofc,self.twat_restr)
        if dmove > 2.5:
            self.features['wat_fofc_ref_xrs'],self.features['wat_fofc_ref_hier'],dmove = self.prsr(wpdb_in,whier_in,wxrs_in,peak_object.shaped_map_fofc,self.twat_restr)

        self.features['wat_cc_fofc_out']=self.rscc(self.features['wat_fofc_ref_hier'],self.features['wat_fofc_ref_xrs'],"resname HOH",ref_object.fofc_map_data)
        self.features['wat_fofc_coord_out'] = self.features['wat_fofc_ref_hier'].atoms()[0].xyz

        if write_pdb:
            ref_object.write_pdb_file(self.features['wat_in_hier'],peak['db_id']+"_w_refin.pdb")
            ref_object.write_pdb_file(self.features['wat_fofc_ref_hier'],peak['db_id']+"_fofc_"+str(outstr)+"_w_refout.pdb")

        #2FOFC, start from 2fofc refined position
        self.features['wat_2fofc_ref_xrs'],self.features['wat_2fofc_ref_hier'],dmove = self.prsr(wpdb_in,self.features['wat_fofc_ref_hier'],self.features['wat_fofc_ref_xrs'],
                                                                                                 self.target_2fofc,self.lwat_restr)
        if dmove > 1.5:
            self.features['wat_2fofc_ref_xrs'],self.features['wat_2fofc_ref_hier'],dmove = self.prsr(wpdb_in,self.features['wat_fofc_ref_hier'],self.features['wat_fofc_ref_xrs'],
                                                                                                     self.target_2fofc,self.twat_restr)

        self.features['wat_cc_2fofc_out']=self.rscc(self.features['wat_2fofc_ref_hier'],self.features['wat_2fofc_ref_xrs'],"resname HOH",ref_object.twofofc_map_data)
        self.features['wat_2fofc_shift'] = self.xform_to_original(whier_in,self.features['wat_2fofc_ref_hier'],self.peak_coord)
        pptbx.renumber_residue(self.features['wat_2fofc_shift'],self.resid)

        if write_pdb:
            ref_object.write_pdb_file(self.features['wat_2fofc_ref_hier'],peak['db_id']+"_2fofc_"+str(outstr)+"_w_refout.pdb")
            ref_object.write_pdb_file(self.features['wat_2fofc_shift'],peak['db_id']+"_2fofc_"+str(outstr)+"_origW_refout.pdb")

        #SO4 REFINEMENT AGAINST INVERTED DENSITY
        #first calculate CC for refined position against the inverted maps
        self.features['so4_cc_fofc_inv_in']=self.rscc(self.features['so4_fofc_ref_hier'],self.features['so4_fofc_ref_xrs'],"resname SO4",ref_object.fofc_inv_map_data)
        self.features['so4_cc_2fofc_inv_in']=self.rscc(self.features['so4_2fofc_ref_hier'],self.features['so4_2fofc_ref_xrs'],"resname SO4",ref_object.twofofc_inv_map_data)

        #then refine against inverted map, tight restraints
        self.features['so4_ifofc_ref_xrs'],self.features['so4_ifofc_ref_hier'],dmove = self.prsr(spdb_in,self.features['so4_fofc_ref_hier'],self.features['so4_fofc_ref_xrs'],
                                                                                                 ref_object.fofc_inv_map_data,self.tso4_restr)
        self.features['so4_cc_fofc_inv_out']=self.rscc(self.features['so4_ifofc_ref_hier'],self.features['so4_ifofc_ref_xrs'],"resname SO4",ref_object.fofc_inv_map_data)

        #calculate CC from coordinates refined against inv map, but to original 2fofc map
        self.features['so4_cc_fofc_inv_rev']=self.rscc(self.features['so4_ifofc_ref_hier'],self.features['so4_ifofc_ref_xrs'],"resname SO4",ref_object.fofc_map_data)

        #repeat for 2fofc
        self.features['so4_i2fofc_ref_xrs'],self.features['so4_i2fofc_ref_hier'],dmove = self.prsr(spdb_in,self.features['so4_2fofc_ref_hier'],self.features['so4_2fofc_ref_xrs'],
                                                                                                   ref_object.twofofc_inv_map_data,self.tso4_restr)
        self.features['so4_cc_2fofc_inv_out']=self.rscc(self.features['so4_i2fofc_ref_hier'],self.features['so4_i2fofc_ref_xrs'],"resname SO4",ref_object.twofofc_inv_map_data)
        self.features['so4_cc_2fofc_inv_rev']=self.rscc(self.features['so4_i2fofc_ref_hier'],self.features['so4_i2fofc_ref_xrs'],"resname SO4",ref_object.twofofc_map_data)

        #WAT CHECK AGAINST INVERTED DENSITY
        #calculate inv CC's for water, no refinement as only one atom which should be correctly positioned
        self.features['wat_cc_fofc_inv']=self.rscc(self.features['wat_fofc_ref_hier'],self.features['wat_fofc_ref_xrs'],"resname HOH",ref_object.fofc_inv_map_data)
        self.features['wat_cc_2fofc_inv']=self.rscc(self.features['wat_2fofc_ref_hier'],self.features['wat_2fofc_ref_xrs'],"resname HOH",ref_object.twofofc_inv_map_data)
 

    def ref_contacts(self):
        ppctx = CctbxHelpers()
        w_coord = self.features['wat_2fofc_shift'].atoms()[0].xyz
        self.features['w_contacts']=ppctx.contacts_to_coord(w_coord,self.struct_data.strip_pdb_hier,self.struct_data.orig_symmetry)
        s_contacts=[]
        s_coords  = (atom.xyz for atom in self.features['so4_2fofc_shift'].atoms())
        for coord in s_coords:
            c_cont = ppctx.contacts_to_coord(coord,self.struct_data.strip_pdb_hier,self.struct_data.orig_symmetry) 
            for c_dict in c_cont:
                s_contacts.append(c_dict)
        s_contacts.sort(key = lambda x: x['distance'])
        self.features['s_contacts'] = s_contacts

 

    def rotations(self,peak_object,ref_object,write_pdb=False):
        cc_f_sum,cc_f_sq_sum,cc_2_sum,cc_2_sq_sum=0.0, 0.0, 0.0, 0.0
        for axis in (1,2,3,4):
            rotated_fofc = self.rotate_so4(self.features['so4_fofc_ref_hier'],axis,60)
            rotated_2fofc = self.rotate_so4(self.features['so4_2fofc_ref_hier'],axis,60)
            if write_pdb:
                rotated_fofc.write_pdb_file(self.features['db_id']+"_rot60_f"+str(axis)+".pdb")
                rotated_2fofc.write_pdb_file(self.features['db_id']+"_rot60_2"+str(axis)+".pdb")
            #hack to get this back into cctbx?
            rotf_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(rotated_fofc.as_pdb_string()))
            rot2f_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(rotated_2fofc.as_pdb_string()))
            rotf_xrs = rotf_pdb.xray_structure_simple(crystal_symmetry=peak_object.symmetry)
            rot2f_xrs = rot2f_pdb.xray_structure_simple(crystal_symmetry=peak_object.symmetry)
            cc60_f = self.rscc(rotated_fofc,rotf_xrs,"resname SO4",ref_object.fofc_map_data)
            cc60_2 = self.rscc(rotated_2fofc,rot2f_xrs,"resname SO4",ref_object.twofofc_map_data)
            cc_f_sum = cc_f_sum + cc60_f
            cc_f_sq_sum = cc_f_sq_sum + cc60_f**2
            cc_2_sum = cc_2_sum + cc60_2
            cc_2_sq_sum = cc_2_sq_sum + cc60_2**2
        meanf = cc_f_sum/4.0
        mean2 = cc_2_sum/4.0
        self.features['so4_fofc_mean_cc60']=  meanf #average cc for all 4 rotations
        self.features['so4_2fofc_mean_cc60']=  mean2
        self.features['so4_fofc_stdev_cc60'] = np.sqrt(cc_f_sq_sum/4.0 - meanf**2) #std
        self.features['so4_2fofc_stdev_cc60'] = np.sqrt(cc_2_sq_sum/4.0 - mean2**2)
                
    def rotate_so4(self,pdb_hier,axis,degrees):
        #selects atom pairs for rotation, atom 1 is central S or P, atom 2 is an oxygen
        s_select = "(name S) or (name P)"
        if axis == 1:
            o_select = "name O1"
        if axis == 2:
            o_select = "name O2"
        if axis == 3:
            o_select = "name O3"
        if axis == 4:
            o_select = "name O4"
        atoms = pdb_hier.atoms()  
        sel_cache = pdb_hier.atom_selection_cache()
        s_atsel = sel_cache.selection(s_select)
        s_atom = atoms.select(s_atsel)
        #gets the sulfur/phosphorous atom
        s_coords = list(s_atom.extract_xyz())[0]
        o_atsel = sel_cache.selection(o_select)
        o_atom = atoms.select(o_atsel)
        o_coords = list(o_atom.extract_xyz())[0]
        return self.rotate_about_2ptaxis(pdb_hier,s_coords,o_coords,degrees)


    def rotate_about_2ptaxis(self,pdb_hier,coor1,coor2,rot):
        # function that rotates a set of coordinates around an arbitrary
        # axis defined by two points by a rotation in degrees
        #coor1 is the center of rotation origin (input as 3tuple)
        #coor2 gives the vector which forms the rotation axis
        #a new copy of coords are returned as hierarchy
        new_hier = pdb_hier.deep_copy()	
        x1,y1,z1 = coor1
        x2,y2,z2 = coor2
        rad = 2*np.pi*rot/360.0 
        (u,v,w) = (x2-x1,y2-y1,z2-z1)
        L = u**2 + v**2 + w**2
        rt = matrix.rt(([(u**2 + (v**2+w**2)*math.cos(rad))/L,
                         (u*v*(1 - math.cos(rad)) - w*math.sqrt(L)*math.sin(rad))/L,
                         (u*w*(1-math.cos(rad)) + v*math.sqrt(L)*math.sin(rad))/L,
                         (u*v*(1-math.cos(rad))+w*math.sqrt(L)*math.sin(rad))/L,
                         (v**2 + (u**2+w**2)*math.cos(rad))/L,
                         (v*w*(1-math.cos(rad))-u*math.sqrt(L)*math.sin(rad))/L,
                         (u*w*(1-math.cos(rad))-v*math.sqrt(L)*math.sin(rad))/L,
                         (v*w*(1-math.cos(rad))+u*math.sqrt(L)*math.sin(rad))/L,
                         (w**2 + (u**2+v**2)*math.cos(rad))/L],
                        [((x1*(v**2+w**2) - u*(y1*v + z1*w))*(1-math.cos(rad))+(y1*w - z1*v)*math.sqrt(L)*math.sin(rad))/L,
                         ((y1*(u**2+w**2) - v*(x1*u + z1*w))*(1-math.cos(rad))+(z1*u - x1*w)*math.sqrt(L)*math.sin(rad))/L,
                         ((z1*(u**2+v**2) - w*(x1*u + y1*v))*(1-math.cos(rad))+(x1*v - y1*u)*math.sqrt(L)*math.sin(rad))/L,]))
        atoms = new_hier.atoms()
        sites = atoms.extract_xyz()
        atoms.set_xyz(rt.r.elems * sites + rt.t.elems)
        return new_hier



    def prsr(self,pdb_in,hier_in,xrs_in,target_map,restraints):
        start_coord = hier_in.atoms()[0].xyz
        xrs_out,hier_out = self.rsrefine(pdb_in,hier_in,xrs_in,target_map,restraints)
        end_coord =  hier_out.atoms()[0].xyz
        dmove = np.linalg.norm(np.subtract(start_coord,end_coord))
        return xrs_out,hier_out,dmove

    def rscc(self,hier,xrs,selection,map):
        cc = self.calc_rscc(hier,xrs,selection,map)
        return cc

    def calc_rt(self,coords1,coords2):
        """
        takes two sets of coords as numpy arrays, each row
        being one point in R3 (pdb atom).  Must have 1:1
        correspondance between atoms for both sets
        no sanity checking!
        computes com, centers coordinates, uses svd
        to generate 3x3 rotation matrix for 1-->2
        also outputs translation vector between com
        adapted from https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        """
        com_1 = np.nanmean(coords1,axis=0)
        com_2 = np.nanmean(coords2,axis=0)
        cent1 = np.subtract(coords1,com_1[None,:])
        cent2 = np.subtract(coords2,com_2[None,:])
        s = np.dot(cent1.T,cent2) #pseudo cov matrix
        U,S,vH = np.linalg.svd(s)
        R = np.dot(vH.T,U.T)
        det =np.linalg.det(R)#must be +/- 1
        i3 = np.eye(3)
        i3[-1,-1] = det #trick to avoid det -1 reflections
        R = np.dot(vH.T,np.dot(i3,U.T))
        #R = np.dot(U,vH)
        trans = com_2 - np.dot(R,com_1.T).T
        return R,trans

    def coord_rt(self,coords,rotation_matrix,translation_vector):
        #expects coords as rows, columns are x,y,z
        #out = Rx + t
        R = rotation_matrix
        crot = np.dot(R,coords.T).T
        ctran = crot + translation_vector[None,:]
        return ctran

    def xform_to_original(self,hier1,hier2,origin_shift):
        #takes two sets of coordinates refined in peak reference setting
        #returns coords2 in initial pdb setting as hierarchy
        coords1 = []
        coords2 = []
        for atom in hier1.atoms():
            coords1.append(atom.xyz)
        for atom in hier2.atoms():
            coords2.append(atom.xyz)
        npcoords1 = np.array(coords1)
        npcoords2 = np.array(coords2)
        if npcoords1.shape[0] == 1: #single atom = water
            trans = np.subtract(npcoords2,npcoords1)
            coords_in_pdb = origin_shift + trans
            new_hier = copy.deepcopy(hier1)
            new_hier.atoms()[0].set_xyz(coords_in_pdb[0])
        else:
            rmatrix,trans = self.calc_rt(npcoords1,npcoords2)
            rt_coords = self.coord_rt(npcoords1,rmatrix,trans)
            shift_to_pdb = np.array(origin_shift) - 5.0
            coords_in_pdb = rt_coords + shift_to_pdb[None,:]
            new_hier = copy.deepcopy(hier1)
            for index,atom in enumerate(new_hier.atoms()):
                atom.set_xyz(coords_in_pdb[index])
        return new_hier

        

