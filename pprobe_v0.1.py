from __future__ import division
# LIBTBX_SET_DISPATCHER_NAME phenix.pprobe_test

import sys,os,time,copy
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
from iotbx import file_reader
from iotbx import phil
from iotbx import reflection_file_utils
from iotbx import crystal_symmetry_from_any
from mmtbx.monomer_library import pdb_interpretation
from mmtbx.monomer_library import server
from mmtbx import map_tools
from mmtbx import find_peaks
from cctbx import maptbx
from cctbx.array_family import flex
import iotbx.pdb 
from libtbx import easy_pickle
from cStringIO import StringIO

import mmtbx.utils
from libtbx.utils import Sorry
from libtbx.utils import multi_out


from PProbe_tasks import PProbeTasks
from PProbe_tasks import PhenixTasks
from PProbe_dataio import DataIO
from PProbe_output import Output

ppio = DataIO()
null_log = open(os.devnull,'w')

legend = """
TBA

"""
master_params_str = """\
include scope libtbx.phil.interface.tracking_params
input {
  pdb {
    model_pdb = None
      .type = path
      .short_caption = Model file
      .multiple = False
      .optional = False
      .help = Model file name
      .style = file_type:pdb bold input_file
    strip_pdb = None
      .type = path
      .optional = False
      .short_caption = Stripped (no solvent)
      .multiple = False
      .help = Model file with solvent removed
      .style = file_type:pdb bold input_file
    peaks_pdb = None
      .type = path
      .short_caption = PDB file of peaks
      .multiple = False
      .optional = False
      .help = Peak list in PDB format (e.g. chain 'A' from phenix.find_peaks_holes)
      .style = file_type:pdb bold input_file
  }
  reflection_data {
    include scope mmtbx.utils.data_and_flags_str
    reflection_file_name = None
      .type = path
      .optional = False
      .short_caption = Input SF Data
      .help = File with experimental data (most of formats: CNS, SHELX, MTZ, etc).
      .style = file_type:hkl bold input_file process_hkl child:fobs:data_labels \
             child:rfree:r_free_flags_labels child:d_min:high_resolution \
             child:d_max:low_resolution
  }
  input_map {
    map_coeff_file = None
      .type = path
      .multiple = False
      .short_caption = MAP maps (no solvent)
      .help = File with 2fo-fc and fo-fc maps pre-computed
    map_diff_label = "FOFCWT,PHFOFCWT"
      .type = str
      .short_caption = column label for FOFCWT/PHFOFCWT coefficients
    map_real_label = "2FOFCWT,PH2FOFCWT"
      .type = str
      .short_caption = column label for 2FOFCWT/PH2FOFCWT coefficients
  }
  model_param {
    model_dict_file = None
       .type = path
       .multiple = False
       .short_caption = file with stat model for classifying
       .help = usually python dict as txt with all coeff "pprobe_master.dict"
  }
  data_pkl {
    peak_dict = None
       .type = path
       .multiple = False
       .optional = True
       .short_caption = input pkl from previous run of extract
  }
  parameters {
    score_res = None
      .type = float
      .short_caption = input resolution for peak scoring
    map_omit_mode = *omitsw omitsol valsol asis 
      .type = choice
      .short_caption = atom omits during map generation
      .help = Options for atom omits during map generation or allowing mapin
    peak_pick_cutoff = 3.0
      .type = float
      .short_caption = sigma cutoff in FOFC for identifying peaks
    write_maps = True
     .type = bool
     .short_caption = Save map coefficients
    write_peaks = True
     .type = bool
     .short_caption = Write peaks to PDB file if not input
    write_strip = False
     .type = bool
     .short_caption = Write stripped PDB 
  }
}    
maps{
  include scope mmtbx.maps.map_and_map_coeff_params_str
  map_coefficients {
    map_type = 2mFo-DFc
    format = mtz
    mtz_label_amplitudes = 2FOFCWT
    mtz_label_phases = PH2FOFCWT
    fill_missing_f_obs = False
  }
  map_coefficients {
    map_type = mFo-DFc
    format = mtz
    mtz_label_amplitudes = FOFCWT
    mtz_label_phases = PHFOFCWT
    fill_missing_f_obs = False 
  }
  scattering_table = wk1995  it1992  *n_gaussian  neutron electron
    .type = choice
    .help = Choices of scattering table for structure factors calculations
  bulk_solvent_correction = True
    .type = bool
  anisotropic_scaling = True
    .type = bool
  skip_twin_detection = False
    .type = bool
    .short_caption = Skip automatic twinning detection
    .help = Skip automatic twinning detection
}
peak_search{
  include scope mmtbx.find_peaks.master_params
}
output {
  directory = None
    .type = path  
  output_file_name_prefix = None
    .type = str
    .short_caption = Output prefix
  output_peak_prefix = peak
    .type = str
    .short_caption = label for peak data (must be 4 characters)
}
pprobe{
  write_pkl = True
    .type = bool
    .short_caption = save .pkl files at various processing stages
    .help = saves .pkl files of all data and parameters
  write_hist = True
    .type = bool
    .short_caption = create a plot of peaks on ED/CC grid
  extract = True
    .type = bool
    .short_caption = extract features or read pkl and rescore
  dev{
    set_chain = None
      .type = str
      .short_caption = Set new chainid for output peaks
    renumber = False
      .type = bool
      .short_caption = Force new peak renumbering
    write_ref_pdb = False
      .type = bool
      .short_caption = Write PDBs for every peak
      .help = Dev feature, produces many files for every peak (caution!)
    pdb_out_str = None
      .type = str
      .short_caption = string for pdb/map writing (cev feature)
    ressig = None
      .type = float
      .short_caption = user input restraint for realspace 
      .help = dev option -- not well tested
    write_maps = False
      .type = bool
      .short_caption = Write CCP4 maps for every peak
      .help = Dev feature, produces many files for every peak (caution!)
    write_pkl = True
      .type = bool
      .short_caption = save .pkl files at various processing stages
      .help = saves .pkl files of all data and parameters
  }
}
gui
  .help = "GUI-specific parameter required for output directory" 
  {
  output_dir = None
    .type = path
    .style = output_dir
}

"""

#adjust cautiously, much of the functionality of pprobe relies on 
#having a range of peaks, even with short contacts

peak_search_param_str="""\
peak_search{
  map_next_to_model {
    min_model_peak_dist = 1.3
    max_model_peak_dist = 8.0
    min_peak_peak_dist = 1.5
  }
  max_number_of_peaks = 10000
  peak_search {
    peak_search_level = 1
    max_peaks = 0
    min_distance_sym_equiv = None
    general_positions_only = False
    min_cross_distance = 1.8
    min_cubicle_edge = 5
  }
}
"""

def run(args):
  all_params = process_inputs(args)
  run_pprobe(all_params)


def run_pprobe(all_params, log = sys.stdout):
  print >> log, "-"*79
  print >> log, "PPROBE PEAK ANALYSIS:"

  ppout = Output()
  ppio = DataIO()
  outstem = all_params.output.output_file_name_prefix
  
  omit_mode = all_params.input.parameters.map_omit_mode
  pptask = PhenixTasks(phenix_params=all_params.output.output_file_name_prefix+"_pprobe.param")

  if all_params.pprobe.extract:
    peak_unal_db=pptask.feature_extract()
    if all_params.pprobe.write_pkl:
      filename = all_params.output.output_file_name_prefix+"_peakdb.pkl"
      easy_pickle.dump(filename,peak_unal_db)
  else:
    pkl_file = all_params.input.data_pkl.peak_dict
    print >> log, "   --> reading feature data from %s" % pkl_file
    peak_unal_db = easy_pickle.load(pkl_file)
    null_sanity(peak_unal_db,all_params)
    null_peak = peak_unal_db[-8861610501908601326] #null_peak
    omit_mode = null_peak['info'].get('omit_mode','omitsw')
  input_feat=list(pdict for pdict in peak_unal_db.values() if ( pdict['model'] == 4 and pdict['status'] == 0))
  ori_feat=list(pdict for pdict in peak_unal_db.values() if pdict['model'] == 3 and pdict['status'] == 5)
  master_array=ppio.extract_raw(input_feat)

  if len(ori_feat) > 0:
    assp_array = ppio.extract_raw(ori_feat)
  else:
    assp_array = None


  #here, switch task classes to avoid conflicts with deep cctbx things
  print >> log, "-"*79
  pptask = PProbeTasks()
  pptask.data_process(master_array)
  pptask.post_process(peak_unal_db,master_array,input_feat)

  if assp_array is not None:
    pptask.data_process(assp_array)
    pptask.post_process(peak_unal_db,assp_array,ori_feat)

  pptask.validate_peaks(peak_unal_db)
  #if all_params.pprobe.write_hist:
  #  pptask.data_vs_hist(master_array,outstr=all_params.output.output_file_name_prefix)


  print >> log,"-"*79
  print >> log,"Analyzing Peak Data . . . "

  ppout.initialize_lists(peak_unal_db,omit_mode=omit_mode)
  ppout.breakdown(peak_unal_db)
  ppout.write_ref_pdb(peak_unal_db)
  report_file=all_params.output.output_file_name_prefix+"_report.log"
  print >> log,"-"*79
  print >> log,"Done! Writing peak report to %s" % report_file+"_report.log"
  ppout.assemble_report(peak_unal_db)
  print >> log,"OUTPUT FILES:"
  print >> log,"  Results/Report:                    %s" % outstem+"_report.log"
  print >> log,"  PProbe Solvent Model as PDB:       %s" % outstem+"_solvent_model.pdb"
  print >> log,"  Refined S/W coords as PDB:         %s" % outstem+"_refined_coords.pdb"
  print >> log,"  Input Params for future reference: %s" % outstem+"_pprobe.param"
  print >> log,"DONE!"


def process_inputs(args, log = sys.stdout):
  print >> log, "-"*79
  print >> log, "PProbe RUN at %s" % time.ctime()
  print >> log, "Processing all Inputs:"
  #process phils in order to not overwrite inputs with defaults
  #phil from above
  master_phil = phil.parse(master_params_str, process_includes=True)
  #map params from phenix defaults (phil)
  maps_phil = phil.parse(mmtbx.maps.map_and_map_coeff_params_str)
  search_phil = phil.parse(peak_search_param_str)

  #merge phil objects?
  total_phil = master_phil.fetch(sources=[maps_phil,search_phil])

  #inputs is somehow different -- object with specific params and lists of files
  #process after all phil?
  inputs = mmtbx.utils.process_command_line_args(args = args, master_params = total_phil)

  #params object contains all command line parameters
  working_phil = inputs.params
  params = working_phil.extract()

  #check for master param dictionary
  if params.input.model_param.model_dict_file is None:
    params.input.model_param.model_dict_file = "pprobe_master.dict"
  if not os.path.isfile(params.input.model_param.model_dict_file):
    print >> log,"WARNING -- param file not found!"
    print >> log,"--> trying pprobe_master.dict . . . "
    params.input.model_param.model_dict_file = "pprobe_master.dict"
    if not os.path.isfile(params.input.model_param.model_dict_file):
      raise Sorry("Master Param Dictionary %s not found!" % params.input.model_param.model_dict_file)

  if params.pprobe.extract:
    #check for proper PDB input
    #count up PDB files found
    pdb_count = len(inputs.pdb_file_names)
    for pdbin in (params.input.pdb.model_pdb,
                  params.input.pdb.strip_pdb,
                  params.input.pdb.peaks_pdb):
      if pdbin is not None:
        pdb_count = pdb_count + 1
    if (pdb_count == 1) and (len(inputs.pdb_file_names)==1):
      #one vanilla pdb to be used as model
      params.input.pdb.model_pdb = inputs.pdb_file_names[0]
    elif (pdb_count == 3) and (len(inputs.pdb_file_names)==0):
      pass #three explicit PDBs, hopefully correct
    else:
      raise Sorry("\n\tInput 1 PDB for automatic stripping and peak finding \n"+\
                  "\tor all PDB files specifically, like so: \n"+\
                  "\tfor explicit input: \n"+\
                  "\t\tmodel_pdb=XXX.pdb strip_pdb=YYY.pdb peaks_pdb=ZZZ.pdb \n"+\
                  "\tfor automatic pdb generation: \n"+\
                  "\t\tXXX.pdb")


    #check for proper reflection file input
    reflection_files = inputs.reflection_files
    if (len(reflection_files) == 0):
      raise Sorry("Reflection data or map coefficients required")
    if (len(reflection_files) > 1):
      raise Sorry("Only one type of reflection data can be entered \n"+\
                  "Enter map coefficients with map_coeff_file=XXX.mtz \n"+\
                  "or structure factor files as XXX.(any supported)")
    else:
      params.input.reflection_data.reflection_file_name = reflection_files[0].file_name()

    #filename setup
    model_basename = os.path.basename(params.input.pdb.model_pdb.split(".")[0])
    if (len(model_basename) > 0 and params.output.output_file_name_prefix is None):
      params.output.output_file_name_prefix = model_basename
    if params.input.input_map.map_coeff_file is not None:
      params.input.parameters.write_maps = False
    new_params =  master_phil.format(python_object=params)
    #okay, see if we're where we want to be
    print >> log, "Runtime Parameters:"
    new_params.show()


    #DATA PROCESSING  
    #setup model pdb (required and should be known)
    crystal_symmetry = check_symmetry(inputs,params,log)
    model_pdb_input = iotbx.pdb.input(file_name = params.input.pdb.model_pdb)
    model_hier = model_pdb_input.construct_hierarchy()
    model_hier.remove_hd()
    model_xrs = model_hier.extract_xray_structure(crystal_symmetry = crystal_symmetry)

    #strip pdb if needed,write result
    if (params.input.pdb.strip_pdb is None) and (params.input.parameters.map_omit_mode != "asis"):
      strip_xrs,strip_hier = create_strip_pdb(model_hier,model_xrs,params.input.parameters.map_omit_mode,log)
      strip_filename = params.output.output_file_name_prefix+"_pprobe_strip.pdb"
      print >> log, "Writing Strip PDB to: ",strip_filename
      strip_hier.write_pdb_file(file_name = strip_filename,crystal_symmetry=crystal_symmetry,append_end=True,anisou=False)
      params.input.pdb.strip_pdb = strip_filename
    elif params.input.parameters.map_omit_mode == "asis":
      strip_xrs,strip_hier = model_xrs,model_hier
      params.input.pdb.strip_pdb = params.input.pdb.model_pdb
    else:
      strip_pdb_input = iotbx.pdb.input(file_name = params.input.pdb.strip_pdb)
      strip_hier = strip_pdb_input.construct_hierarchy()
      strip_hier.remove_hd()
      strip_xrs = strip_hier.extract_xray_structure(crystal_symmetry = crystal_symmetry)


    #Make maps if map_coefficients not input,write out by default
    if (params.input.input_map.map_coeff_file is None):

      hkl_in = file_reader.any_file(params.input.reflection_data.reflection_file_name,force_type="hkl")
      hkl_in.assert_file_type("hkl")
      reflection_files = [ hkl_in.file_object ]
      f_obs,r_free_flags = setup_reflection_data(inputs,params,crystal_symmetry,reflection_files,log)
      #maps object is list of miller arrays
      maps = create_pprobe_maps(f_obs,r_free_flags,params,strip_xrs,strip_hier,log)
      map_fname =params.output.output_file_name_prefix+"_pprobe_maps.mtz" 
      print >> log, "Writing PProbe maps to MTZ file: ",map_fname
      maps.write_mtz_file(map_fname)
      params.input.input_map.map_coeff_file = params.output.output_file_name_prefix+"_pprobe_maps.mtz"
    else:
      print "READING MAP FILE: ",params.input.input_map.map_coeff_file
      #setup input map coefficients

      map_coeff = reflection_file_utils.extract_miller_array_from_file(
        file_name = params.input.input_map.map_coeff_file,
        label     = params.input.input_map.map_diff_label,
        type      = "complex",
        log       = null_log)

      if params.input.parameters.score_res is None:
        params.input.parameters.score_res = map_coeff.d_min()
        print >> log, "  Determined Resolution Limit: %.2f" % params.input.parameters.score_res
        print >> log, "    -->Override with \"score_res=XXX\""
      map_fname = params.input.input_map.map_coeff_file


    # if peaks not input, find and write to pdb
    if params.input.pdb.peaks_pdb is None:
      if params.input.parameters.map_omit_mode != "valsol":
        peaks_result = find_map_peaks(params,strip_xrs,log)
        pdb_str = peaks_pdb_str(peaks_result)
        peak_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(pdb_str))
        peak_hier = peak_pdb.construct_hierarchy()
        peak_filename =params.output.output_file_name_prefix+"_pprobe_peaks.pdb" 
        print >> log,"Writing Peaks to %s:" % peak_filename
        peak_hier.write_pdb_file(file_name = peak_filename,crystal_symmetry=crystal_symmetry,append_end=True,anisou=False)
        params.input.pdb.peaks_pdb = peak_filename
      else:
        peak_filename =params.output.output_file_name_prefix+"_pprobe_peaks.pdb" 
        peak_xrs,peak_hier = create_sol_pdb(model_hier,model_xrs,params.input.parameters.map_omit_mode,log)
        print >> log,"Writing Peaks to %s:" % peak_filename
        peak_hier.write_pdb_file(file_name = peak_filename,crystal_symmetry=crystal_symmetry,append_end=True,anisou=False)
        params.input.pdb.peaks_pdb = peak_filename

    #Wrap up, display file names and info for manual input
    #save parameters for next stage
    new_phil = working_phil.format(python_object = params)
    phil_fname = params.output.output_file_name_prefix+"_pprobe.param" 
    f = open(phil_fname, "w")
    f.write(new_phil.as_str())
    f.close()
    print >> log, "_"*79
    print >> log, "Inputs Processed, final files:"
    print >> log, "   Model PDB: ",params.input.pdb.model_pdb
    print >> log, "   Strip PDB: ",params.input.pdb.strip_pdb
    print >> log, "   Peaks PDB: ",params.input.pdb.peaks_pdb
    print >> log, "   Map Coeff: ",map_fname
    print >> log, "   Resolution: %.2f" % params.input.parameters.score_res
    print >> log, "   Params: ",phil_fname
    #also return params
    return params
  else: #only rescoring from pkl
    #filename setup
    pkl_basename = os.path.basename(params.input.data_pkl.peak_dict.split(".")[0])
    if (len(pkl_basename) > 0 and params.output.output_file_name_prefix is None):
      params.output.output_file_name_prefix = pkl_basename
      pkl_file = params.input.data_pkl.peak_dict
    if not os.path.isfile(pkl_file):
      raise Sorry("\n\tPKL input requested but no file available\n"+\
                  "\t\t\t cannot find %s" % pkl_file)
    new_phil = working_phil.format(python_object = params)
    phil_fname = params.output.output_file_name_prefix+"_pprobe.param" 
    f = open(phil_fname, "w")
    f.write(new_phil.as_str())
    f.close()

    new_params =  master_phil.format(python_object=params)
    print >> log, "Runtime Parameters:"
    new_params.show()
    return params
      
def find_map_peaks(params,strip_xrs,log):
  #Adapted from mmtbx fine_peaks.py and
  #phenix find_peaks_holes.py, simplified to just give
  #coords and map levels for clustered peaks in difference map
  print >> log, "_"*79
  print >> log, "Finding Difference Map Peaks:"
  map_coeff = reflection_file_utils.extract_miller_array_from_file(
    file_name = params.input.input_map.map_coeff_file,
    label     = params.input.input_map.map_diff_label,
    type      = "complex",
    log       = null_log)
  peaks_result = find_peaks.manager(
    fmodel=fake_fmodel(strip_xrs),
    map_type= None,
    map_cutoff=params.input.parameters.peak_pick_cutoff,
    params=params.peak_search,
    use_all_data = True,
    map_coeffs=map_coeff,
    log=log)
  peaks_result.peaks_mapped()#cluter/arrange found peaks?
  peaks = peaks_result.peaks()#returns heights,coords(frac)
  unit_cell = strip_xrs.unit_cell()#need cell for cartesian
  peaks.sites = unit_cell.orthogonalize(peaks.sites)
  
  return peaks


class fake_fmodel(object):
  #peak picking somehow needs an fmodel
  #which we may not have if map coefficients input directly
  #fake class to return r values to indicate real data
  #and a reference to an xrs
  def __init__(self,xrs):
    self.xray_structure=xrs
  def r_work(self):
    return 0.1
  def r_free(self):
    return 0.1

def peaks_pdb_str(peaks):
  #hack to put together a pdb string
  pdb_str=""
  serial = 1
  for peak, xyz in zip(peaks.heights,peaks.sites):
    x,y,z = xyz
    b = peak
    #chain is set to "P" for peak
    pdb_str = pdb_str+format_atom(serial,"O","","HOH","P",serial,"",x,y,z,1.0,b,"O","")
    serial = serial + 1
  return pdb_str


def format_atom(serial,name,alt,resname,chain,resid,ins,x,y,z,occ,temp,element,charge):
  #canonical PDB formatting
  pdb_fmt_str ="{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n" 
  return pdb_fmt_str.format("ATOM",serial,name,alt,resname,chain,resid,ins,x,y,z,occ,temp,element,charge)  

  
def check_symmetry(inputs,params,log):
  #check for usable and consistent symmetry
  #somehow, this happens beforehand sometimes?
  #inputs finds a symmetry object from somewhere?
  print >> log, "_"*79
  print >> log,"Checking Crystal Symmetry:"
  crystal_symmetry = None
  crystal_symmetry = inputs.crystal_symmetry 
  if (crystal_symmetry is None):
    crystal_symmetries = []
    for f in [str(params.input.pdb.model_pdb), str(params.input.reflection_data.reflection_file_name)]:
      cs = crystal_symmetry_from_any.extract_from(f)
      if(cs is not None): 
        crystal_symmetries.append(cs)
    if(len(crystal_symmetries) == 1): 
      crystal_symmetry = crystal_symmetries[0]
    elif(len(crystal_symmetries) == 0):
      raise Sorry("No crystal symmetry found.")
    else:
      if(not crystal_symmetries[0].is_similar_symmetry(crystal_symmetries[1])):
        raise Sorry("Crystal symmetry mismatch between different files.")
      crystal_symmetry = crystal_symmetries[0]
  print >> log,"  Unit Cell: ",crystal_symmetry.unit_cell().parameters()
  return crystal_symmetry

def setup_reflection_data(inputs,params,crystal_symmetry,reflection_files,log):
  #setup reflection data
  f_obs, r_free_flags = None, None
  rfs = reflection_file_utils.reflection_file_server(
    crystal_symmetry = crystal_symmetry,
    force_symmetry   = True,
    reflection_files = reflection_files,
    err              = StringIO())
  parameters = mmtbx.utils.data_and_flags_master_params().extract()
  if (params.input.reflection_data.labels is not None):
    parameters.labels = params.params.input.reflection_data.labels
  if (params.input.reflection_data.r_free_flags.label is not None):
    parameters.r_free_flags.label = params.input.reflection_data.r_free_flags.label
  try:
    determined_data_and_flags = mmtbx.utils.determine_data_and_flags(
      reflection_file_server = rfs,
      parameters             = parameters,
      keep_going             = True,
      log                    = StringIO())
    f_obs = determined_data_and_flags.f_obs
    r_free_flags = determined_data_and_flags.r_free_flags
  except:
    print "DATA PROCESSING ERROR --> picking first miller array"
    f_obs = rfs.miller_arrays[0]
  if (params.input.reflection_data.labels is None):
    params.input.reflection_data.labels = f_obs.info().label_string()
  if (params.input.reflection_data.reflection_file_name is None):
    params.input.reflection_data.reflection_file_name = parameters.file_name
  assert f_obs is not None
  print >> log, "_"*79
  print >> log,  "Input data:"
  print >> log, "  Iobs or Fobs:", f_obs.info().labels
  if (r_free_flags is not None):
    print >> log, "  Free-R flags:", r_free_flags.info().labels
    params.input.reflection_data.r_free_flags.label = r_free_flags.info().label_string()
  else:
    print >> log, "  Free-R flags Not present, generating ..."
    r_free_flags = f_obs.generate_r_free_flags(fraction=0.05, max_free=1000)
  # Merge anomalous if needed
  if (f_obs.anomalous_flag()):
    sel = f_obs.data()>0
    f_obs = f_obs.select(sel)
    merged = f_obs.as_non_anomalous_array().merge_equivalents()
    f_obs = merged.array().set_observation_type(f_obs)
    r_free_flags = r_free_flags.select(sel)
    merged = r_free_flags.as_non_anomalous_array().merge_equivalents()
    r_free_flags = merged.array().set_observation_type(r_free_flags)
    f_obs, r_free_flags = f_obs.common_sets(r_free_flags)

  if params.input.parameters.score_res is None:
    params.input.parameters.score_res = f_obs.d_min()
    print >> log, "  Determined Resolution Limit: %.2f" % params.input.parameters.score_res
    print >> log, "    -->Override with \"score_res=XXX\""
  return f_obs,r_free_flags

def split_mac_sol(pdb_hier,pdb_xrs,omit_mode,log):
  #modify pdb structure with selected solvent omits
  input_hier = pdb_hier
  input_xrs = pdb_xrs
  atom_selection_manager = input_hier.atom_selection_cache()
  if(omit_mode is not 'asis'):
    watnames = ['HOH',]      #other water names, TIP etc?
    so4names = ['SO4','PO4']
    othnames = ['DTT', 'MAL', 'EOH', 'SUC', 'SCN', 'P6G', 'GSH', 'CO3', 'CIT', 'BOG', 'NO3', 'IMD', 'BME', 'ACY', 
                'PGE', 'PG4', 'TRS', 'MPD', 'DMS', 'PEG', 'ACT', 'EDO', 'GOL', 'CL', ' BR', 'AZI', 'GNP', 'BGC', 
                'BEN', 'H4B', 'SF4', 'GLC', 'RET', '1PE', 'ACP', 'CAC', 'FLC', 'EPE', 'AKG', 'LDA', 'SAM', 'POP', 
                'F3S', 'NAI', 'MLI', 'NDG', 'THP', 'HED', 'NH4', 'TLA', 'FES', 'HEC', 'MRD', 'UNL', 'IPA', 'PLP', 
                'MES', 'NCO', 'PLM', 'MAN']
    elenames = ['BR','I','CO','CD','NI','CU','FE','MN','CA','CL','MG','ZN']
    if omit_mode == 'omitsw':      
      to_omit = watnames + so4names
      print >> log, "  Omitting SO4/PO4 and HOH from model"
    if omit_mode == 'omitsol' or omit_mode == 'valsol':
      to_omit = list(set(watnames + so4names + othnames + elenames))
      print >> log, "  Omitting %d common solvent molecules (e.g. HOH,SO4,CIT,BME,EDO,Cl,...)" % len(to_omit)
    omit_sel_str = "resname "+" or resname ".join(to_omit).format(['"{:3s}"'*len(to_omit)])
    omit_selection = atom_selection_manager.selection(string = omit_sel_str)
    return omit_selection



def create_strip_pdb(pdb_hier,pdb_xrs,omit_mode,log):
  print >> log, "_"*79
  print >> log, "Creating PDB stripped of selected solvent species:"
  #modify pdb structure with selected solvent omits
  if(omit_mode is not 'asis'):
    omit_selection = split_mac_sol(pdb_hier,pdb_xrs,omit_mode,log)
    strip_selection = ~omit_selection
    n_selected = strip_selection.count(True)
    n_omitted = omit_selection.count(True)
    print >> log, "     Omitted %d atoms total" % n_omitted
    strip_xrs = pdb_xrs.select(selection = strip_selection)
    strip_hier = pdb_hier.select(strip_selection)
    return strip_xrs,strip_hier

def create_sol_pdb(pdb_hier,pdb_xrs,omit_mode,log):
  sol_sel = split_mac_sol(pdb_hier,pdb_xrs,omit_mode,log)
  n_selected = sol_sel.count(True)
  print >> log, "     Output %d solvent atoms total" % n_selected
  sol_xrs = pdb_xrs.select(selection = sol_sel)
  sol_hier = pdb_hier.select(sol_sel)
  return sol_xrs,sol_hier


def create_pprobe_maps(f_obs,r_free_flags,params,strip_xrs,strip_hier,log):
  print >> log, "_"*79
  print >> log, "Creating Maps:"
  print >> log, "  Note, R-factors are with selected atoms removed!\n"
  f_obs = f_obs.resolution_filter(
    d_min = params.input.reflection_data.high_resolution,
    d_max = params.input.reflection_data.low_resolution)
  if (r_free_flags is not None):
    r_free_flags = r_free_flags.resolution_filter(
      d_min = params.input.reflection_data.high_resolution,
      d_max = params.input.reflection_data.low_resolution)
  fmodel = mmtbx.utils.fmodel_simple(
    xray_structures         = [strip_xrs],
    scattering_table        = params.maps.scattering_table,
    f_obs                   = f_obs,
    r_free_flags            = r_free_flags,
    outliers_rejection      = True,
    skip_twin_detection     = False,
    bulk_solvent_correction = True,
    anisotropic_scaling     = True)
  fmodel_info = fmodel.info()
  fmodel_info.show_rfactors_targets_scales_overall(out = log)
  map_obj = mmtbx.maps.compute_map_coefficients(fmodel = fmodel,
                                            params = params.maps.map_coefficients,
                                            pdb_hierarchy = strip_hier,
                                            log = log)
  return map_obj



def null_sanity(peak_unal_db,all_params):
  #previous versions stored a pkl without a null peak, which contains important parameters
  #hack to create a dummpy peak with appropriate info
  null_peak = peak_unal_db.get(-8861610501908601326,None)
  if null_peak is None:
    for option in all_params.input.parameters.map_omit_mode:
      if option[0] == "*":
        omit_mode = option[1::]
    model_pdb = all_params.input.pdb.model_pdb[0]
    cs = crystal_symmetry_from_any.extract_from(model_pdb)

    null_peak = {'edc': 0, 'sol_contacts': [], '2fofc_sig_out': 0.0, 'wat_2fofc_ref_oricoords': (), 'pick_name': '', 
                 'c1': 0.0, 'clust_score': 0, 'so4_cc_2fofc_inv_out': 0.0, 'ambig': 0, 'charge': 0.0, 'bin': 0, 
                 'wat_cc_2fofc_in': 0.0, 'llgS': 0.0, 'fofc_sig_in': 0.0, '2fofc_sig_in': 0.0, 'llgW': 0.0, 'scr3': 0.0, 
                 'modexp_clust': [], 'tflag': 0, 'orires': '', 'so4_cc_fofc_inv_in': 0.0, 'resid': '0', 'resat': 'NUL_P0_O', 
                 'coord': (999.99, 999.99, 999.99), 'dmove': 0.0, 'unrg': 6626794411521655213, 'wat_cc_fofc_in': 0.0, 'ptype': '', 
                 'cont_db': {}, 'batch': 0, 'wat_cc_fofc_inv': 0.0, 'mod_for': [], 'pdb_code': '', 'clash': False, 
                 'so4_fofc_stdev_cc60': 0.0, 'contacts': [], 'so4_2fofc_stdev_cc60': 0.0, 'wat_cc_2fofc_out': 0.0, 'label': 0, 
                 'strip_contacts': [], 'score': 0.0, 'clust_rank': 0, 'rc': 0, 'so4_fofc_coord_out': (), 'prob': 0.0, 'wat_fofc_coord_out': (), 
                 'db_id': 'null_P_00000', 'ori_chain': 'P', 'so4_cc_2fofc_out': 0.0, 'wm': 0, 'wl': 0, 'solc': 0.0, 'so4_cc_fofc_out': 0.0, 
                 'wt': 0, 'master_dict': {}, 'ol': 0, 'so4_cc_fofc_in': 0.0, 'oh': 0, 'clust_mem': [], 'so4_cc_2fofc_in': 0.0, 'omit': 0, 
                 'pick': 0, 'cscore': 0.0, 'so4_cc_fofc_inv_rev': 0.0, 'peak_contacts': [], 'cc': 0, 'mm_contacts': [], 
                 'unat': -6006398286768061262, 'clust_cent': 0.0, 'cllgW': 0.0, 'cllgS': 0.0, 'unal': -8861610501908601326, 
                 'ori_resid': '0', 'mod_cont': [], 'om': 0, 'anc_for': [], 
                 'so4_2fofc_ref_oricoords': [('X', (0.0, 0.0, 0.0)), ('X', (0.0, 0.0, 0.0)), ('X', (0.0, 0.0, 0.0)), 
                                             ('X', (0.0, 0.0, 0.0)), ('X', (0.0, 0.0, 0.0))], 
                 'filter_mask': [0, 0, 0, 0], 'info': {'symmetry': cs,
                                                       'param': all_params,
                                                       'omit_mode': omit_mode}, 
                 'scr1': 0.0, 'scr2': 0.0, 's_contacts': [], 'so4_cc_2fofc_inv_in': 0.0, 'clust_pair': [], 'anchor': {}, 
                 'chainid': 'P', 'w_contacts': [], 'so4_cc_fofc_inv_out': 0.0, 'so4_cc_2fofc_inv_rev': 0.0, 'mf': 0, 
                 'so4_2fofc_mean_cc60': 0.0, 'anc_cont': [], 'wat_cc_2fofc_inv': 0.0, 
                 'prob_data': array([[ 0.,  0.,  0.,  0.],
                                     [ 0.,  0.,  0.,  0.],
                                     [ 0.,  0.,  0.,  0.]]), 
                 'fofc_sigo_scaled': 0.0, 'fofc_sig_out': 0.0, 'chiS': 0.0, 'chiW': 0.0, 'vol_fofc': 0.0, 'cchiW': 0.0, 
                 'cchiS': 0.0, 'peak_unal_db': {}, 'status': -1, 'inputs': {'ori_hier': None}, 
                 'mflag': 0, 'warnings': [], 'omit_contacts': [], '2fofc_sigo_scaled': 0.0, 'sol_mod': [], 'fc': 0, 'worst_mm': {}, 
                 'sp': 0, 'wat_cc_fofc_out': 0.0, 'st': 0, 'sm': 0, 'sl': 0, 'so4_fofc_mean_cc60': 0.0, 'model': 5, 'resolution': 0.0, 'vol_2fofc': 0.0}
    peak_unal_db[-8861610501908601326] = null_peak
    return
  else:
    null_peak['info']['param'] = all_params
    return


if (__name__ == "__main__"):
  run(args=sys.argv[1:])
