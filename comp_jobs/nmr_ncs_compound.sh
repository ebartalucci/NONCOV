#!/bin/bash
# Author: Ettore Bartalucci
# Date  : September 2024
#
# *********************************************** DESCRIPTION *****************************************************
# Perform NMR and Natural Chemical Shift Analysis calculations for each of the optimixed xyz files in the folder
#
# Use same level of theory for both
# 
# 

# *********************************************** OUTPUT FILE *****************************************************
# nmr_ncs_multi.inp

# ------ Variables to adjust ---------------------------------------------------
moldir=$(pwd)                     		# Current folder with XYZ files
charge=0                        		# Molecular charge
mult=1	                          		# Spin multiplicity
nmr_ncs_method="pbe0 pcSseg-2 RIJK defgrid3 nososcf printgap NMR"
multijob_inp="nmr_ncs_multi"         # Output file for geometry optimization jobs
newj=New_Step
endj=Step_End

# Initialize the input files
echo "" > ${multijob_inp}.inp

# ------ Loop over all .xyz files in the directory -----------------------------

for xyz_file in $moldir/*.xyz; do
  
  mol_name=$(basename "$xyz_file" .xyz)

  # Add ORCA job for NMR Calculation
  cat << EOF >> ${multijob_inp}.inp
# NMR Calculation for $mol_name

$newj

! $nmr_method

* xyzfile $charge $mult ${mol_name}.xyz

%maxcore 8000

%basis
  auxJK "AutoAux"
end

%nbo
  NBOKeyList = "$NBO NCS=0.01,I,U,XYZ $END"
end

%pal
  nprocs 8
end

%eprnmr
  Ori = GIAO
  giao_1el = giao_1el_analytic
  giao_2el = giao_2el_rijk
  nuclei = all { shift }
end

$endj

EOF

done

echo "Compound Job written to ${multijob_inp}.inp"
