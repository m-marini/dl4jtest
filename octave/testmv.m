F = [
  0.95 0.0 0.0;
  0.05 0.5 0.5;
]

VAR = {
  eye(2) * 4,
  eye(2) * 0.2,
};

P = [64 64];
C = [-1 -1; 1 1];

[ X Y ] = meshgrid(
  (0 : P) * (C(2, 1) - C(1, 1)) / P(1) + C(1, 1),
  (0 : P) * (C(2, 2) - C(1, 2)) / P(2) + C(1, 2));

#VAR = [0.2 0.05; 0.05 0.1]
 
function Z = normbivar(X, Y, VAR)
  [n, m] = size(X);
  XX = [X(:) Y(:)];
  D = zeros(size(XX, 1), 1);
  size(XX)
  for i = 1 : length(D)
    D(i) = XX(i, :) * inv(VAR) * XX(i, :)';
  endfor
  Z =exp(-D/2) / 2 / pi / sqrt(det(VAR));
  Z = reshape(Z, n, m);
endfunction

#Z = +0*gauss(X,Y, [ 0 0], 3) + 0*gauss(X,Y, [ 0.5 0.5], 0.1) + gauss(X,Y, [ 0.5 -0.5], 0.01);
n = size(F, 1);
PR = cell(n);
for i = 1 : n
  PR{i} = F(i, 1) * normbivar(X - F(i, 2), Y - F(i, 3), VAR{i});
endfor

#Z = 2 * double(P{3} > P{1}+P{2}) + double(P{2} > P{1}+P{3});
Z = -PR{1}+PR{2};
surfc(X,Y,Z);
#contourf(X,Y,Z, -1:0.1:1);
