"""
TensorView.py 0.9beta 


Working in Python 3.6.5


packages used: Numpy


By: Corbin Lewis & Leonard Mueller 


For work making use of TensorView.py, please cite the following article: 


R.P. Young, C.R. Lewis, C. Yang, L. Wang, J.K. Harper, and L.J. Mueller, "TensorView.nb:  A Software Tool for Displaying NMR Tensors," Magnetic Resonance in Chemistry, XXX(XX), YYY-YYY (2018)


This program is provided by the author "as is" with no expressed or implied warranty or assumption of liability for damages resulting from the use of this software.

"""

#Import packages and modules
import numpy as np
from os import system, name
from time import sleep

def userInput():
	"""
	userInput function: take in user input for formatting output lines and output file 
	
	Input
	-----
	User provided input in the form of string or int
	
	Returns
	-------
	s
		scaling factor for tensor 
	atomNumber
		atom serial number from pdb file/user input
	atomName
		atom name from pdb file/user input
	seqNum
		atom residue sequence number from pdb file/user input
	atomType
		element of the given atom from pdb file/user input
	fileName
		file name for the output file, with ".txt" extension 
	"""
	#Define default vaules for user input
	defaultNum = int(1)
	defaultStr = "C"

	#Clear screen
	clear()

	#Call for user to define pdb values, scale factor and file name for output
	s=float(input("Enter scaled length of largest principal component for display (angstroms) (default = 1): ") or "1")
	atomNumber=int(input('Enter atom serial number (default = 1): ') or "1")
	atomName=(input('Enter atom serial name (C, Ca, O, N) (default = C): ') or "C")
	seqNum=int(input('Enter atom residue sequence number (default = 1): ') or 1)
	atomType=(input('Enter the element of atom (C, O, N) (default = C): ') or "C")
	fileName=(input('Enter a file name without fle extension (default = Out): ') or "Out")
	fileName=fileName+".txt"

	#Clear screen
	clear()
	return[s, atomNumber, atomName, seqNum, atomType, fileName];
	
def wait():
	"""
	wait function: print out a string line, wait for user keypress and clears the terminal window using the clear function

	Returns
	-------
	Clears screen by calling clear function
	"""
	input("Press Enter to continue...")
	clear()
	return;


def clear():
	# If Windows
    if name == 'nt':
        system('cls')
 
    # If Linux or Mac
    else:
        system('clear')
	


def buildMat():
	"""
	buildMat function: build up a 3x3 matrix from user input for each ij matrix element

	Input
	-----
	User provided input in the form of string
	
	Returns
	-------
	array
		example array to guide user input
	nuMatrix
		matrix populated with User's input in numpy matrix format 
	"""
	#Define matrix and example array
	matrix=[]
	array=[['A11','A12','A13'],['A21','A22','A23'],['A31','A32','A33']]
	
	#Take in user data for matrix and print out input
	print("Enter the matrix elements following the order shown below")
	print('\n')
	for row in array:
 	 print(' '.join(map(str,row)))
	print('\n')
	for i in range(3):
		matrix.append([])
		for j in range(3):
			elem=input('Enter element A'+str(i+1)+str(j+1)+'(default = 1): ') or 1
			elem=float(elem)
			matrix[i].append(elem)
	print('\n')
	print("Matrix as entered:")
	nuMatrix=np.matrix(matrix)
	print(nuMatrix)
	print('\n')
	
	#Wait for user to continue
	wait()
	return nuMatrix;		


def symMat(nuMatrix):
	"""
	symMat function: take in a given matrix, transpose matrix and calculate the symmetric matrix form

	Paramaters
	----------
	nuMatrix: int, numpy matrix type
		Matrix containing user input

	Returns
	-------
	symm
		symmetric matrix created from user defined matrix 
	"""
	#Transpose matrix
	transpose=np.transpose(nuMatrix)
	
	#Form symmetric matrix
	symm=np.array((nuMatrix+transpose)/2)
	
	#Print out matrix
	print("Symmetric matrix:")
	print(symm)
	return symm;


def eigenV(symm):
	"""
	eigenV function: take in symmetric matrix, calculate eigenvalues/vectors of the symmetric matrix, compute a matrix with eigenvaules in the diagonal, output values and matrix product

	Paramaters
	----------
	symm: int, numpy array type
		symmetric matrix created from user defined matrix 

	Returns
	-------
	a
		eigenvector from column 0
	b
		eigenvector from column 1
	c
		eigenvector from column 2
	diagonal
		matrix with eigenvalues in the diagonal
	e_vals
		eigenvalue array
	e_valsA
		absolute value of eigenvalue array
	matrixM
		Matrix computed by matrix product between Transposed Eigenvector matrix (TEvM), diagonal matrix, and EvM. (ie. TEvM.Diagonal.EvM) 
	"""
	#Calculate eigenvalues and vectors, take absolute value of eigenvalues 
	e_vals, e_vecs = np.linalg.eig(symm)
	e_valsA=np.absolute(e_vals)
	e_valsAS=np.power(e_valsA,2)
	
	#Compute diagonal matrix, define eigenvector columns as variables and preforme matrix multiplication 
	diagonal=(np.diag(e_valsAS)).round(5)
	a=e_vecs[:, 0].round(5); b=e_vecs[:, 1].round(5); c=e_vecs[:, 2].round(5)
	m_vecs=np.vstack((a,b,c)).round(5)
	m_vecsT=np.transpose(m_vecs).round(5)
	matrixM=m_vecsT.dot(diagonal).dot(m_vecs).round(5)
	
	#Print data
	print('\n')
	print("Eigenvalues:")
	print(e_vals.round(5)),
	print('\n')
	print("Eigenvectors:")
	print(a)
	print('\n')
	print(b)
	print('\n')
	print(c)
	print('\n')
	
	#Wait for user to continue
	wait()
	return [a, b, c, diagonal, e_vals, e_valsA, matrixM];


def prepChi(e_valsA, s, matrixM):
	"""
	prepChi function:  Taking in absolute value of Eigenvalues and Product matrix from eigenV, compute the scaled values given user defined scale (s) and largest abs(eigenvalue) (M)

	Paramaters
	----------
	e_valsA: int, array
		array containing the absolute values of the eigenvalues 
	s: int
		the user defined scale 
	matrixM
		Matrix computed by matrix product between Transposed Eigenvector matrix (TEvM), diagonal matrix, and EvM. (ie. TEvM.Diagonal.EvM) 

	Returns
	-------
	matrixS
		scaled version of matrixM matrix
	"""
	#Define variables 
	q=np.amax(e_valsA)
	M=np.power(q,2)
	sS=np.power(s,2)
	
	#Compute scale based off max eigenvalue and s
	matrixS=(matrixM*sS*(10000/M)).round(5)
	return matrixS;


def format(matrixS, atomNumber, atomName,seqNum, atomType, fileName):
	"""
	format function:  Taking in user defined variables and scaled matrix, output formatted ANISO PDB lines, both to terminal window and to a user defined txt file

	Paramaters
	----------
	matrixS: int, array/matrix
		scaled version of matrixM matrix
	atomNumber: int, (default 1)
		atom number from pdb file
	atomName: srt, (default C)
		the name of the atom of interest
	seqNum: int, (default 1)
		sequence number from pdb file
	atomType: srt, (default C)
		element type of the atom
	fileName: srt, (default Out)
		name for output file

	Returns
	-------
	fileName
		"fileName".txt, formatted line printed out to txt file
	"""
	#Define pdb record type
	type="ANISOU"
	
	#Print out data and formatted line 
	print("Matrix values:")
	print("Uxx:", matrixS.item(0),"Uyy:", matrixS.item(4),"Uzz:", matrixS.item(8),"Uxy:", matrixS.item(1),"Uxz:", matrixS.item(2),"Uyz:", matrixS.item(5))
	uxx=int(matrixS.item(0)); uyy=int(matrixS.item(4)); uzz=int(matrixS.item(8)); uxy=int(matrixS.item(1)); uxz=int(matrixS.item(2)); uyz=int(matrixS.item(5))
	print('\n')
	print("Matrix values after rounding and formatting, copy this line to pdb, below atom of interest:")
	print("                                {:6s} {:6s} {:6s} {:6s} {:6s} {:6s}    ".format("Uxx","Uyy","Uzz","Uxy","Uxz","Uyz"))
	print("{:6s}{:5d} {:^4s}      {:4d}   {:6d} {:6d} {:6d} {:6d} {:6d} {:6d}      {:>2s}".format(type,atomNumber,atomName,seqNum,uxx,uyy,uzz,uxy,uxz,uyz,atomType))
	
	#Save data to file
	with open(fileName, "w") as text_file:
    		text_file.write("{:6s}{:5d} {:^4s}      {:4d}   {:6d} {:6d} {:6d} {:6d} {:6d} {:6d}      {:>2s}".format(type,atomNumber,atomName,seqNum,uxx,uyy,uzz,uxy,uxz,uyz,atomType))
	print ('\n')	
	print("Formatted text is also printed out in file:", fileName)
	print ('\n')
	
	#Wait for user to continue
	wait()
	return;

#Call function to gather user inputs, define output as a series of variables  	
s, atomNumber, atomName, seqNum, atomType, fileName=userInput() 

#Call function to build matrix from user information, define output as nuMatrix 
nuMatrix=buildMat()

#Define symm as the returned array from symMat function call
symm = symMat(nuMatrix)

#Call eigenV function and define output as a series of variables
a, b, c, diagonal, e_vals, e_valsA, matrixM = eigenV(symm)

#Define matrixS as the returned matrix product from prepChi function call
matrixS = prepChi(e_valsA, s, matrixM)

#Call funtion to generate the formatted line and formatted text document
format(matrixS, atomNumber, atomName,seqNum, atomType, fileName)



