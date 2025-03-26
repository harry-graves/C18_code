h = 1;
K = [1, 0, 2; 
     0, 2, 1; 
     0, 0, 1];

Rt = [0, 1, 0, 0; 
      0, 0, 1, -h; 
      1, 0, 0, 4*h];

% Define 3D world points in homogeneous coordinates
X = [0, 0, 0, 1;  % X1
     0, 1, 0, 1;  % X2
     0, 1, 1, 1;  % X3
     0, 0, 1, 1;  % X4
     1, 0, 1, 1;  % X5
     1, 1, 1, 1]; % X6

% Compute projected 2D image points
for i = 1:size(X, 1)
    x_hom = K * Rt * X(i, :)'; % Compute homogeneous image coordinates
    x_proj = x_hom ./ x_hom(3); % Convert to inhomogeneous coordinates
    disp(['Projected point ', num2str(i), ': (', num2str(x_proj(1)), ', ', num2str(x_proj(2)), ')']);
end