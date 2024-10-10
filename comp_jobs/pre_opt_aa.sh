#!/bin/bash
# Author: Ettore Bartalucci
# Date  : September 2024
#
# *********************************************** DESCRIPTION *****************************************************
# Perform unconstrained Geometry Optimization with high structural accuracy for each of the xyz files in the folder
#
# 
# 
# 

# *********************************************** OUTPUT FILE *****************************************************
# aa_geoopt_pre.inp

# ------ Variables to adjust ---------------------------------------------------
moldir=$(pwd)                     		# Current folder with XYZ files
charge=0                        		# Molecular charge
mult=1	                          		# Spin multiplicity
geoopt_method="r2scan-3c def2-mTZVPP def2/J D4 OPT TIGHTSCF TIGHTOPT NOAUTOSTART"
multijob_inp="aa_geoopt_pre"  		        # Output file for geometry optimization jobs
newj=New_Step
endj=Step_End

# Initialize the input files
echo "" > ${multijob_inp}.inp

# ------ Loop over all .xyz files in the directory -----------------------------

for xyz_file in $moldir/*.xyz; do
  
  mol_name=$(basename "$xyz_file" .xyz)

  # Add ORCA job for Geometry Optimization
  cat << EOF >> ${multijob_inp}.inp

# Tightly accurate Geometry Optimization for $mol_name

$newj

! $geoopt_method

* xyzfile $charge $mult ${mol_name}.xyz

%maxcore 4000

%pal
  nprocs 8
end

$endj

EOF

done

echo "Compound Job written to ${multijob_inp}.inp"
