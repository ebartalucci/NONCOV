#!/bin/bash
# Author: Ettore Bartalucci
# Date  : September 2024
#
# *********************************************** DESCRIPTION *****************************************************
# Perform constrained Geometry Optimization and subsequent NMR calculations for each of the xyz files in the folder
#
# Step 1. Constrained Geometry Optimization only relaxing light nuclei
# Step 2. NMR calculations
# Step 3. Save all geoopt jobs in one file and all nmr jobs in another file

# *********************************************** OUTPUT FILE *****************************************************
# compound_calc.inp

# ------ Variables to adjust ---------------------------------------------------
moldir=$(pwd)                     # Current folder with XYZ files
charge=0                          # Molecular charge
mult=1                          # Spin multiplicity
geoopt_method="r2scan-3c def2-mTZVPP def2/J D4 OPT TIGHTSCF"
nmr_method="pbe0 pcSseg-2 RIJK defgrid3 nososcf printgap NMR Largeprint"
multijob_inp="benzene_H2O_relaxH_multi"          # Output file for geometry optimization jobs

# Initialize the input files
echo "" > ${multijob_inp}.inp

i=1

# ------ Loop over all .xyz files in the directory -----------------------------

for xyz_file in $moldir/*.xyz; do
  
  mol_name=$(basename "$xyz_file" .xyz)

  # Add ORCA job for Geometry Optimization
  cat << EOF >> ${multijob_inp}.inp
# Constrained Geometry Optimization for $mol_name

! $geoopt_method

* xyzfile $charge $mult ${mol_name}.xyz

%maxcore 4000

%geom
  optimizehydrogens true  # Constrained geometry optimization for light nuclei
end

%pal
  nprocs 8
end

$new_job
EOF

  # Add ORCA job for NMR Calculation
  cat << EOF >> ${multijob_inp}.inp
# NMR Calculation for $mol_name

! $nmr_method

* xyzfile $charge $mult ${multijob_inp}_job${i}.xyz

%maxcore 10000

%basis
  auxJK "AutoAux"
end

%pal
  nprocs 8
end

%eprnmr
  Ori = GIAO
  giao_1el = giao_1el_analytic
  giao_2el = giao_2el_same_as_scf
  nuclei = all { shift, ssall }
end

$new_job
EOF

  i=$((i+1))

done

echo "Compound Job written to ${multijob_inp}.inp"
