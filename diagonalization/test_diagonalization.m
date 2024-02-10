% Define the new matrix
matrix_variable_new = [-126.142, -0.344, 3.750;
                       -3.781, -96.355, 24.604;
                       1.691, 12.833, -113.089];

% Compute eigenvectors and eigenvalues
[V_new, D_new] = eig(matrix_variable_new);

% Display the diagonalized form
disp("Diagonalized Matrix:")
disp(D_new)
