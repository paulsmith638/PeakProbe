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
import cProfile

class CctbxHelpers:
    """
    Class of PProbe functions that dig deep into cctbx
    Watch our for later scipy conflicts
    """
    def __init__(self):
        self.pput = Util()

    def do_cprofile(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats()
        return profiled_func


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

    #@do_cprofile
    def contacts_to_all(self,pdb_hier,symmetry,cutoff=6.0):
        #finds all contacts within cutoff in an entire pdb_hier
        xrs = pdb_hier.extract_xray_structure(crystal_symmetry=symmetry)
        asu_mappings = xrs.asu_mappings(buffer_thickness=cutoff+1.0)
        pair_generator =  crystal.neighbors_fast_pair_generator(asu_mappings,distance_cutoff = cutoff)
        natoms = pdb_hier.atoms().size()
        #distance matrix impractical as we may have multiple contacts to the same atom by symmetry
        #all of which need to be preserved
        all_cont = {}
        for i in range(natoms):
            all_cont[i] = []
        for pair in pair_generator:
            pi = pair.i_seq
            pj = pair.j_seq
            ps = pair.j_sym
            pd = int(np.sqrt(pair.dist_sq)*1000)/1000.0 #avoid base2 float issues for uniquifying
            all_cont[pi].append((pj,pd,ps)) #need both from and to (across asu)
            all_cont[pj].append((pi,pd,ps))
        ncont = 0

        
        awl_db = {} # refs to awl objects to pass
        for awl in pdb_hier.atoms_with_labels():
            awl_db[awl.i_seq] = awl
        
        all_cont_db = {}

        for iseq,clist in all_cont.iteritems():
            #print "PROTO",iseq,clist
            ncont = ncont+len(clist)
            source_at = iseq
            source_awl = awl_db[source_at]
            uni_c = set(clist)
            # for each source awl, generate list of unique tuples (cont awl,distance,sym)
            cpairs = list((awl_db[c_at[0]],c_at[1],c_at[2]) for c_at in uni_c )
            contacts,s_unique_id = self.get_contacts(source_awl,cpairs,cutoff=cutoff)
            all_cont_db[s_unique_id] = contacts
        print "      Found %d contact pairs" % ncont
        return all_cont_db

    #@do_cprofile
    def contacts_to_coord(self,coord,pdb_hier,symmetry,cutoff=6.0):
        """
        Used to use extract map_model, but that caused problems
        This function takes cartesian coordinate, pdb_hier, and symmetry and returns
        sorted list of dictionaries, each a particular contact

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
        nmodels = len(pdb_hier.models())
        if nmodels == 1:
            comb = orig_str+dummy_atom
        else:
            newmodel = nmodels + 1
            comb = orig_str+"MODEL %s\n" % newmodel
            comb = comb+dummy_atom
            comb = comb+"ENDMDL\n" 
        dummy_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(comb))
        dummy_hier = dummy_pdb.construct_hierarchy()
        atsel = "chain ZZ and resid 9999"
        dummy_select = dummy_hier.atom_selection_cache().selection(string = atsel)
        atsel_index = np.argwhere(dummy_select.as_numpy_array())
        dummy_xrs = dummy_pdb.xray_structure_simple(crystal_symmetry = symmetry)
        #find neighbors with symmetry -- use pair_finding with asu_mappings
        #doing this for each peak may be a bad idea, use all_contacts version instead if possible
        asu_mappings = dummy_xrs.asu_mappings(buffer_thickness=cutoff)
        pair_generator =  crystal.neighbors_fast_pair_generator(asu_mappings,distance_cutoff = cutoff)
        
        peak_vector_list = []
        #neighbor_mask = pair_generator.neighbors_of(atsel_arr)
        #neighbors = pair_generator.select(neighbor_mask)
        for pair in pair_generator:
            if pair.i_seq == atsel_index:
                #our peak is first atom, but we want pairs in both directions
                #to provide exhaustive list of contacts
                #store index number and difference vector and sym to make unique
                #we don't care which symop, just how far away
                rdist = int(np.sqrt(pair.dist_sq)*1000)/1000.0 #avoid float error
                peak_vector_list.append((pair.j_seq,rdist,pair.j_sym))
            if pair.j_seq == atsel_index:
                rdist = int(np.sqrt(pair.dist_sq)*1000)/1000.0 
                peak_vector_list.append((pair.i_seq,rdist,pair.j_sym))

        unique = set(peak_vector_list)
        dummy_atoms = dummy_hier.atoms()

        #selection is boolean flex array, sizeof no atoms, can be indexed directly
        #get awl for our dummy atom
        s_awl = dummy_atoms.select(dummy_select)[0].fetch_labels()
        #list((awl_db[c_at[0]],c_at[1],c_at[2]) for c_at in uni_c )
        #next, get list of contacts (awl,dist)
        selection=dummy_hier.atom_selection_cache().selection("not all")
        if len(unique) == 0:
            return []
        cont_list = []
        for conti in unique:
            selection[conti[0]] = True
            sel_awl = dummy_atoms.select(selection)[0].fetch_labels()
            cont_list.append((sel_awl,conti[1],conti[2])) #pass source awl, index for target, distance
            selection[conti[0]] = False #unselect
        contacts,s_unal = self.get_contacts(s_awl,cont_list,cutoff=cutoff)
        return contacts


    def merge_hier(self,hier_list,symmetry):
        pdb2str = lambda hier: hier.as_pdb_string(write_scale_records=False, append_end=False, 
                                                  interleaved_conf=0, atom_hetatm=True, sigatm=False, anisou=False, 
                                                  siguij=False, output_break_records=False)

        allpdb = ""
        for index,hier in enumerate(hier_list):
            allpdb = allpdb+"MODEL %s\n" % str(index+1)
            allpdb = allpdb+pdb2str(hier)
            allpdb = allpdb+"ENDMDL\n"
        dummy_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(allpdb))
        dummy_hier = dummy_pdb.construct_hierarchy()
        dummy_hier.remove_hd()
        dummy_hier.atoms_reset_serial()
        #dummy_hier.write_pdb_file('merge.pdb')
        return dummy_hier

    def get_contacts(self,s_atom,cpairs,cutoff=6.0):
        pput = Util()
        s_resname=s_atom.resname.strip()
        s_chain=str(s_atom.chain().id).strip()
        s_model_id = s_atom.model_id.strip()
        s_element=s_atom.element.strip()
        s_name=s_atom.name.strip()
        s_altloc = s_atom.altloc.strip()
        s_resid = s_atom.resseq.strip()
        s_coord = s_atom.xyz
        s_unat,s_unal,s_unrg = pput.gen_unids(s_atom)
        s_resat = s_resname+"_"+s_chain+s_resid+"_"+s_name
        contacts = []
        for cpair in cpairs:
            distance = cpair[1]
            #contact_atom
            c_atom = cpair[0]
            c_sym = int(cpair[2])
            resname=c_atom.resname.strip()
            chain=str(c_atom.chain().id).strip()
            model_id = c_atom.model_id.strip()
            element=c_atom.element.strip()
            name=c_atom.name.strip()
            altloc = c_atom.altloc.strip()
            resid = c_atom.resseq.strip()
            coord = c_atom.xyz
            unat,unal,unrg = pput.gen_unids(c_atom)
            resat = resname+"_"+chain+resid+"_"+name
            if s_model_id == "":
                s_model_id = 5
            if model_id == "":
                model_id = 5
            special = False
            ctype = "unknown"
            if s_unat == unat:
                ctype = "self"
                if distance < 1.8:
                    special = True
            elif int(model_id) == int(s_model_id):
                ctype = "intra"
            else:
                ctype = "inter"
            #everything goes into a dictionary (some converted to int)
            contact={"name":name,"chain":chain,"element":element,"distance":distance,
                     "coord":coord,"resname":resname,"altloc":altloc,"resid":int(resid),
                     "model":int(model_id),"special":special,"unat":unat,"unal":unal,"unrg":unrg,
                     "s_name":s_name,"s_chain":s_chain,"s_element":s_element,
                     "s_coord":s_coord,"s_resname":s_resname,"s_altloc":s_altloc,"s_resid":int(s_resid),
                     "s_model":int(s_model_id),"ctype":ctype,"s_unat":s_unat,"s_unal":s_unal,"s_unrg":s_unrg,
                     "s_resat":s_resat,"resat":resat,"sym":c_sym}
            contacts.append(contact)
        contacts.sort(key = lambda x: x['distance'])
        return contacts,s_unal
