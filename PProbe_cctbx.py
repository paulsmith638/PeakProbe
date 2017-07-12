from __future__ import division
#GENERIC IMPORTS
import sys,os,re,copy
import numpy as np
#CCTBX IMPORTS
import iotbx.pdb
from iotbx.ccp4_map import write_ccp4_map
from cctbx import crystal
from cctbx import sgtbx
from scitbx.array_family import flex
from PProbe_util import Util


class CctbxHelpers:
    """
    Class of PProbe functions that dig deep into cctbx
    Watch our for later scipy conflicts
    """
    def __init__(self):
        self.pput = Util()

    def write_local_map(self,input_map_data,filename_base,peak_object):
        #writes out map in std setting
        last_grid = peak_object.grid_last
        write_ccp4_map(
            file_name=filename_base+".ccp4",
            unit_cell=peak_object.so4_xrs.unit_cell(),
            space_group=sgtbx.space_group_info("P1").group(),
            gridding_first=(0,0,0),
            gridding_last=(last_grid,last_grid,last_grid),
            map_data=input_map_data,
            labels=flex.std_string(["local_map",]))

    def renumber_residue(self,pdb_hier,num):
        #sets resid for all atoms in hier
        for model in pdb_hier.models():
            for chain in model.chains():
                for rg in chain.residue_groups():
                    rg.resseq = num


    def contacts_to_coord(self,coord,pdb_hier,symmetry,cutoff=6.0):
        """
        Used to use extract map_model, but that caused problems
        This function takes cartesian coordinate, pdb_hier, and symmetry and returns
        sorted extensive dictionay of local contacts including all sym-mates

        very hacked together, jams pdb strings together
        create a dummy pdb, put water at the peak site in the original coordinate system
        then use fast pair generater to get all contacts to our peak atom
        """
        dummy_atom=self.pput.write_atom(1,"O","","HOH","ZZ",9999,"",coord[0],coord[1],coord[2],1.0,35.0,"O","")
        #hack to add an atom to a pdb
        orig_str = pdb_hier.as_pdb_string(write_scale_records=False, append_end=False, 
                                          interleaved_conf=0, atoms_reset_serial_first_value=1, 
                                          atom_hetatm=True, sigatm=False, anisou=False, siguij=False, 
                                          output_break_records=False)
        comb = dummy_atom+orig_str
        dummy_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(comb))
        dummy_hier = dummy_pdb.construct_hierarchy()
        dummy_xrs = dummy_pdb.xray_structure_simple(crystal_symmetry = symmetry)
        #find neighbors with symmetry -- use pair_finding with asu_mappings
        #doing this for each peak may be a bad idea, perhaps the whole structure with peaks at once?
        asu_mappings = dummy_xrs.asu_mappings(buffer_thickness=cutoff)
        pair_generator =  crystal.neighbors_fast_pair_generator(asu_mappings,distance_cutoff = cutoff)
        peak_vector_list = []
        for pair in pair_generator:
              if pair.i_seq == 0:
              #our peak is first atom, but we want pairs in both directions
              #to provide exhaustive list of contacts
              #store index number and difference vector
              #we don't care which symop, just how far away
                    peak_vector_list.append((pair.j_seq,pair.diff_vec))
              if pair.j_seq == 0:
                    peak_vector_list.append((pair.i_seq,pair.diff_vec))


        dummy_atoms = dummy_hier.atoms()
        selection=dummy_hier.atom_selection_cache().selection("not all")
        #selection is boolean flex array, sizeof no atoms, can be indexed directly
        isel = selection.iselection()
        #isel is the index array size_t, empty, useless?
        contact_atom_list = []
        for conti in peak_vector_list:
              selection[conti[0]] = True
              sel_atom = dummy_atoms.select(selection)
              #return tuple with atom_i,atom_ref,vect_dist
              contact_atom_list.append((conti[0],sel_atom,conti[1]))
              selection[conti[0]] = False #unselect
        contacts=[]
        #put all the good stuff in a list for use in identifying original residue
        #and looking at local environment
        for conti in contact_atom_list:
              distance = np.linalg.norm(conti[2])#distance
              if distance < cutoff: #double_check
                    atom = conti[1][0]#what is an af_shared_atom?
                    awl = atom.fetch_labels()
                    resname=awl.resname.strip()
                    chain=str(awl.chain().id).strip()
                    element=awl.element.strip()
                    name=awl.name.strip()
                    altloc = awl.altloc.strip()
                    resid = awl.resseq.strip()
                    special = False
                    #flags coords near special positions
                    #useful for contacts to a water position, not SO4
                    if (resid == '9999' and chain == 'ZZ' and distance < 1.0):
                        special = True
                    contact={"name":name,"chain":chain,"element":element,"distance":distance,
                             "resname":resname,"altloc":altloc,"resid":resid,"special":special}
                    contacts.append(contact)
        contacts.sort(key = lambda x: x['distance'])
        #print "CONTACTS:"
        #for cont in contacts:
        #    print cont
        return contacts


