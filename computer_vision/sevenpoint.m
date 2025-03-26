%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sevenpoint
% is a quick hack to explore recovery of F from just 
% 7 matches
% DWM 1/10/2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

download the other required m functions from 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sevenpoint uses the following functions:
% set_spherical_scene(n_scenepoints,radius)  synthesizes a scene
% set_camera_extrinsics(Xc1,Xt1,cyclo1)      
% set_camera_intrinsics(1,1,0,0,0);
% add_noise(noisepercent/100,x1);            adds noise to image points
% get_centering_matrix(x1);                  statistical centering (see notes)
% f_cubic_coeffs(v,w);                       works out the cubic coeffs (see notes)
% fvec_to_Fmat(f);                           builds matrix from vector
% normalize(C2' * Fc * C1)                   
% crossmatrix(t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% and the following built-ins
% svd(A)         singular value decomposition
% roots(coeffs)  roots of a polynomial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function output = sevenpoint(n_scenepoints,noisepercent)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make the random scene points Xw in a sphere
  radius = 1;
  Xw = set_spherical_scene(n_scenepoints,radius);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the extrinsics ...
% Set cameras 1 and 2 to look at origin of the scene
% from given positions, and with given cyclotorsion.
  Xt1    = [2;1;0];
  Xt2    = [-2;0;0];
  Xc1    = [2;2;-4];
  Xc2    = [-2;2;-4];
  cyclo1 = 0;
  cyclo2 = 0;
  E1     = set_camera_extrinsics(Xc1,Xt1,cyclo1);
  E2     = set_camera_extrinsics(Xc2,Xt2,cyclo2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the intrinsics ...
% f=1, aspect=1, skew=0, u0=0, v0=0 
  K1 = set_camera_intrinsics(1,1,0,0,0);
  K2 = set_camera_intrinsics(1,1,0,0,0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make a 3x4 vanilla projection matrix
% Then build project matrices P1 and P2
  vanilla = [1,0,0,0;0,1,0,0;0,0,1,0];
  P1 = K1*vanilla*E1;
  P2 = K2*vanilla*E2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project the scene points into the images
  x1 = P1*Xw;
  x2 = P2*Xw;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scale 3rd component to unity
% NB: There should be a more matlabby way of doing this!
  for(i=1:3)
    for(j=1:n_scenepoints)
      x1(i,j)=x1(i,j)/x1(3,j);
      x2(i,j)=x2(i,j)/x2(3,j);
    end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add noise to the image positions
  x1 = add_noise(noisepercent/100,x1);
  x2 = add_noise(noisepercent/100,x2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% See Lec notes 3 for statistical centring ...
% Compute the two centering matrices and centre data
% You can check that mean(xc1') gives [0 0 1]
% and std(xc1') gives [1.414 1.414 0]
  C1 = get_centering_matrix(x1);
  C2 = get_centering_matrix(x2);
  xc1 = C1*x1;
  xc2 = C2*x2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NB Comment ...
% After this centering,
% any Fc computed with xc1 and xc2 should be
% converted to the actual F using
% F = C2' * Fc * C1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We have synthesize matching image points,
% so now
% Generate a complete A matrix
% NB!!! Here I am using only the first seven points
% Really I should use my draw=random_draw(7,n_scenepoints) function 
% to choose 7 at random, 
% then replace the i on the RHS (*not* the LHS!) by draw(i)
  for (i=1:7)
    A(i,1) = xc2(1,i) * xc1(1,i);
    A(i,2) = xc2(1,i) * xc1(2,i);
    A(i,3) = xc2(1,i);
    A(i,4) = xc2(2,i) * xc1(1,i);
    A(i,5) = xc2(2,i) * xc1(2,i);
    A(i,6) = xc2(2,i);
    A(i,7) = xc1(1,i);
    A(i,8) = xc1(2,i);
    A(i,9) = 1;
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform SVD to obtain nullspace of A
% (alternative in matlab is to use ker=null(A), then v=ker(:,1) etc
%
  [U,S,V] = svd(A);
  v = V(:,8);
  w = V(:,9);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Work out the coefficients of the cubic (see notes for expression)
  [a0,a1,a2,a3] = f_cubic_coeffs(v,w);
  coeffs=[a3,a2,a1,a0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find a vector of three roots ...
  alpha=roots(coeffs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deal with the three cases ...
  for(i=1:3)
    f = alpha(i)*v + (1-alpha(i))*w;
    Fc = fvec_to_Fmat(f);
    % Remember to decentre the result 
    F = C2' * Fc * C1;
    % normalize the result. F is know only up to scale,
    % so this is superfluous. It is tidy though
    Fnorm = normalize(F);

    % shamefully clumsy code to store the three cases
    if (i==1) 
      F1 = Fnorm;
      sum1=0.0;
      for(j=1:n_scenepoints) 
        dev= x2(:,j)'*F1*x1(:,j); 
        sum1 = sum1 + dev*dev;
      end
    elseif (i==2) 
      F2 = Fnorm;
      sum2=0.0;
      for(j=1:n_scenepoints) 
        dev = x2(:,j)'*F2*x1(:,j); 
        sum2 = sum2 + dev*dev; 
      end
    else
      F3 = Fnorm;
      sum3=0.0;
      for(j=1:n_scenepoints) 
        dev = x2(:,j)'*F3*x1(:,j); 
        sum3 = sum3 + dev*dev; 
      end
    end
  end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% What is the real value of F?
% To use the formula, We need to find [R|t] for the 2nd camera, given that camera 1 is [I|0]
  E= E2*inv(E1);
  for(i=1:3)
    t(i) = E(i,4);
    for(j=1:3)
      R(i,j) = E(i,j);
    end
  end
  Tx = crossmatrix(t);
  K2inv = inv(K2);
  Ftrue = normalize(K2inv' * Tx * R * inv(K2))
  sumtrue=0.0;
  for(j=1:n_scenepoints) 
    dev=x2(:,j)'*Ftrue*x1(:,j); 
    sumtrue = sumtrue + dev*dev;
  end

  output = sqrt([sum1,sum2,sum3,sumtrue]);
   
