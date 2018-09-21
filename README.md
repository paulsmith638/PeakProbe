PeakProbe
=========

Tool for identifying/classifying solvent molecules in macromolecular x-ray crystal structures.

Authors: 
-------
-- Paul Smith, PhD. Assistant Prof. of Chemistry, FordhamUniversity (psmith66@fordham.edu) 
-- Laurel Jones, PhD Candidate,Northwestern University 
-- Michael Tynes, Data Sciences GraduateProgram, Fordham University

Software Prerequisites: 
----
-- Current PHENIX installation (https://www.phenix-online.org/)

Requirements: 
-----
-- A correctly formatted PDB (with or without existing solvent) for a given x-ray structure -- Structure factor data that can be read by PHENIX (cif/mtz/scalepack/etc.) -- All python modules included within this repository (20 in total) -- The PHENIX interface master python file (pprobe\_vXXX.py) 

Usage:
----
-- PeakProbe must be run under the "phenix.python" environment or be made aware of the locations of all necessary CCTBX imports -- A PDB file and structure factor data are required and can be specified in any order or suitable format as per PHENIX norms -- The program has several modes of operation and many configurable parameters, but the essentials are covered below.

_____Basic/Typical usage of PeakProbe:_____

Within a directory containing a clone of this repository, run:

*****phenix.python pprobe_v0.1.py XXXX_input.pdb XXXX_input.mtz > YYYY.log*****

XXXX_input.pdb is a well-formatted PDB file containing a macromolecular structure (with or without solvent)

XXXX_input.mtz contains structure_factor data.  Any format that can be  read by PHENIX should work (.hkl,.scl, etc.).   
Selection of structure factor data (e.g. FOBS, SIGFOBS) etc. should be automatic.  
This file  should not contain any  reflections beyond the useful resolution of the data.  

YYYY.log is a target file for program output.  This log is not particularly useful for non-developers but may be 
helpful should you which to report bugs or other issues to the developers.


Output:
----

PeakProbe typically outputs the following files:
NB: PREFIX will be determined from the PDB filename or can be set at run-time with "output_file_name_prefix=PREFIX"as a command line argument.

>        PREFIX_report.log           --> Description of each peak
>        PREFIX_refined_coords.pdb   --> Debugging only
>        PREFIX_solvent_model.pdb    --> PeakProbe solvent model
>        PREFIX_peakdb.pkl           --> Pkl of feature data
>        PREFIX_pprobe.param         --> Run-time parameters
>        PREFIX_pprobe_peaks.pdb     --> peaks found by PeakProbe
>        PREFIX_pprobe_maps.mtz      --> Calculated Maps
>        PREFIX_pprobe_strip.pdb     --> input structure without solvent

__More detailed descriptions of each file are found below.__


Options:
----
>        The most helpful options for obtaining useful output from PeakProbe are specifying what solvent molecules found in 
>        the input model are removed prior to analysis.  PeakProbe works best when existing solvent is allowed to be omitted.
>        However, specific use cases may make the following options warranted.
>
>
>        Option: omit_mode=ZZZZ 
>            ZZZZ is one of the following:
>                omitsw (default)    -->     HOH and SO4/PO4 are omitted
>                omitsol             -->     Additional common solvent are omitted including PEG,GOL,EDO, etc.
>                valsol              -->     Existing common solvent are ommitted and their coordinates are used as "peaks"
>                                            for analysis -- useful for solvent validation.
>                asis                -->     No atoms are omitted, all solvent present are assumed to be valid and only new
>                                            solvent are considered

___Example:___
>    --To run PeakProbe on an existing structure to validate its current solvent model, use the following:
>        **phenix.python pprobe_v0.1.py XXXX_input.pdb XXXX_input.mtz omit_mode=valsol > YYYY.log**

Detailed Output Descriptions:
-----
PREFIX_report.log:
>        --  Summaries all inputs and outputs to PeakProbe:
>        --  Listing of problem peaks, including likely model errors
>        --  Detailed description of each peak including observed density values, contact distances to all neighbors, peaks, solvent, etc.  
>            Further details include relationships between peaks and existing solvent model atoms, contacts of note (e.g. clashes), 
>            scoring and classification data, and notes about local clusters (e.g. from an unmodeled ligand).  Lastly, the classification and 
>            fate of each peak is output, indicating what species of solvent are likely to exist at a particular peak and what, if anything, 
>            was built at a given peak in the final solvent model.
>       --   Short version of above data, intended for development and data mining.

PREFIX_refined_coords.pdb: 

>       – coordinates for HOH and SO4 placed by real-space refinement, only useful for debugging troublesome or un-converged refinements
            
PREFIX_solvent_model.pdb:

>       -- PDB file containing a probably solvent model.  Note that the contents of this file are specific to the omit_mode being used.
>               omitsw/omitsol      --> solvent in-place of all omitted solvent
>               valsol              --> only validated solvent found in input
>               asis                --> only "new" solvent output
>       -- Chains are as follows, sorted by difference map value:
>               "W"     -->     HOH, water
>               "S"     -->     SO4 (or PO4 if found in existing solvent)
>               "O"     -->     OTH, "other", such as PEG,GOL,CL etc.  If found in the original structure these species are preserved, 
>                               if not, CL is output 
>               "M"     -->     ML1, "metal", identity copied over from existing model if found, otherwise built as "MG"

PREFIX_peakdb.pkl:

>        --     A "pickled" file of the main dictionary used to store all data for all peaks. Currently, feature extraction
>               by PeakProbe is rather slow (requiring ~1sec per peak on a modern CPU).  To bypass the extraction step and 
>               proceed directly to analysis, this file can be input at the command line using the following syntax:
>

___phenix.python pprobe_v0.1.py extract=False data_pkl.peak_dict=PREFIX_peakdb.pkl___

>
>        --     no other PDB or structure factor data are needed and PeakProbe moves directly to analysis

PREFIX_pprobe.param:
>        -- Runtime parameters output by PHENIX, can be edited to tune program functionality and specified at the command line (NB:, not well tested)

PREFIX_pprobe_peaks.pdb:
>            --  PDB file of "peaks" used by PeakProbe, in all but "valsol" modes, these peaks are determined by a peak search of FoFc density 
>                calculated after indicated solvent are omitted.  Default cutoff is 3.0 sigma (unscaled).

PREFIX_pprobe_maps.mtz:
>        --  MTZ file containing weighted structure factors / phases for 2FoFc and FoFc density as generated during PeakProbe 
>            map calculation after indicated solvent are omitted.

PREFIX_pprobe_strip.pdb:
>       -- PDB file following removal of indicated solvent (see omit_mode).

Description:
---
--PeakProbe analyzes the electron density and local atomic environment at a given point within a crystal lattice (a "peak") and assesses a what types of solvent molecules, if any, are likely to occupy that particular point. The program was originally intended to identify rigid polyatomic species, such as sulfate and phosphate, the most common solvent species found in macromolecular structures after water. However, the program will also offer some idea about the possible placement of other common solvent species such as acetate, chloride ion, PEG, etc. as well as metal ions. PeakProbe can be run in various modes, but the essential output of the program is a detailed characterization of the local properties of each "peak" input into or found, including a prediction of what molecules/species should be modeled at that site. The program also outputs a solvent model as a PDB file that can be used directly in refinement procedures. In addition to identifying and placing likely solvent species, PeakProbe also outputs useful data on likely model errors (e.g. incorrect rotamers) found during analysis. The program can be run using data of any resolution, but caution is essential when attempting to model solvent with limiting resolutions below \~3.3A.

A detailed description of the algorithms used and how the model used for solvent classification was trained will be documented in a forthcoming publication and included within the projects source files. In the meantime, feel free to contact Paul Smith (psmith66@fordham.edu) with any questions you have or issues you experience while using PeakProbe.

Caveats:
---
-- Current testing (Sept. 2018) shows that PeakProbe is highly accurate for identifying HOH and SO4/PO4 (as per the program's original intent). However, classifications of other species ("OTH") such as PEG,GOL,CL,etc. and metal ions ("ML1") are less reliable and *ab initio* predictions of these species should be taken as "suggestions". However, if such species are currently placed within an input model and omitted in PeakProbe (see omit\_mode options), the likelihood that such species are valid from a point-of-view of electron density and local contact environment as given by PeakProbe tend to be on point. In other words, if a structure contains glycerol (GOL), but PeakProbe strongly suggests water in place of GOL, a manual inspection of the local electron density is likely to confirm PeakProbe's conclusions.

-- At present, the program can build only water and SO4/PO4 ab initio, but will preserve existing solvent species that the program deems valid. Peaks highly unlikely to be water and highly likely to be occupied by a larger species but that are not currently modeled as such within an input structure will be output in the final solvent model as chloride ions (CL). Similarly, peaks likely to be metal but for which no existing metal model can be found will be output as magnesium (MG). 

-- PeakProbe's output is currently very verbose and includes a great deal of debugging output. The most useful output can be fount in the final report file (XXX\_report.log).

Acknowledgements: 
---
PeakProbe relies upon many CCTBX libraries/tools (https://cci.lbl.gov/cctbx\_docs/) as well as many routines from the associated PHENIX project\ (https://www.phenix-online.org/). The authors thank the developers of these projects for providing such assessable and powerful tools with which to explore. The authors are also indebted to several colleagues within the Fordham University Dept. of Mathematics for their helpful input in navigating much of the statistical and machine-learning landscape necessary to develop PeakProbe.
