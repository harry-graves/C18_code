function calibration

npoints=6;

X=[0,0,0,0,1,1];
Y=[0,1,1,0,0,1];
Z=[0,0,1,1,1,1];

% This should be what you found for the image points
x = [2.00,2.25,2.25,2.00,2.00,2.20];
y = [0.50,0.50,1.00,1.00,1.00,1.00];

% This would add noise to the points
%  x=x+(1-2*rand(1,6))*0.01;
%  y=y+(1-2*rand(1,6))*0.01;


A=zeros(2*npoints,12);

for i=1:npoints
     rx = 2*i-1;
     ry = 2*i;

     A(rx,1) = X(i);
     A(rx,2) = Y(i);
     A(rx,3) = Z(i);
     A(rx,4) = 1;
     A(rx,9)=-X(i)*x(i);
     A(rx,10)=-Y(i)*x(i);
     A(rx,11)=-Z(i)*x(i);
     A(rx,12)=-x(i);

     A(ry,5) = X(i);
     A(ry,6) = Y(i);
     A(ry,7) = Z(i);
     A(ry,8) = 1;
     A(ry,9) =-X(i)*y(i);
     A(ry,10)=-Y(i)*y(i);
     A(ry,11)=-Z(i)*y(i);
     A(ry,12)=-y(i);

end

if npoints==4
  p = null(A);
else
  [U,S,V] = svd(A);
  p=V(:,12);
end


m=1;
for i=1:3
for j=1:4
projectionmatrix(i,j) = p(m);
m=m+1;
end
end
projectionmatrix

Pleft(1,1) = p(1);
Pleft(1,2) = p(2);
Pleft(1,3) = p(3);
Pleft(2,1) = p(5);
Pleft(2,2) = p(6);
Pleft(2,3) = p(7);
Pleft(3,1) = p(9);
Pleft(3,2) = p(10);
Pleft(3,3) = p(11);
Pleftinv = inv(Pleft);

[Q,R] = qr(Pleftinv);

Rot      = inv(Q);
K        = inv(R);

if K(1,1) < 0
   K(1,1) = - K(1,1); % f
   Rot(1,:) = -Rot(1,:);
end
if K(2,2) < 0
   K(2,2) = -K(2,2);
   K(1,2) = -K(1,2);
   Rot(2,:) = -Rot(2,:);
end
if K(3,3) < 0
   K(:,3) = -K(:,3);
   Rot(3,:) = -Rot(3,:);
end

tmp = K(3,3);
P = projectionmatrix/tmp;

intrinsic = K/tmp
Rot
translation = inv(intrinsic)*[P(1,4);P(2,4);P(3,4)]
