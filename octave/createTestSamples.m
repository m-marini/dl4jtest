function [X Y] = createTestSamples(N, F) 
  NF = size(F, 1);
  
  P = zeros(1, NF);
  for i = 1 : NF
    P(i) = F{i}.p;
  endfor

  ## Create expectations
  Y = randpref(P, N);

  ## Create samples
  X = zeros(N, 2);
  for i = 1 : NF
    IDX = find(Y == i);
    M = length(IDX);
    X(IDX, :) = normBivar(M, F{i}.m, F{i}.s);
  endfor
endfunction

function Y = randpref(X,N)
  P = cumulate(X);
  R = rand(N, 1) * P(end);
  Y = zeros(N, 1);
  for i = 1 : N
    Y(i) = find(P >= R(i))(1);
  endfor
endfunction

function Y = cumulate(X) 
  N = length(X);
  Y = zeros(1, N);
  ACC = 0;
  for i = 1 : N
    ACC = ACC + X(i);
    Y(i) = ACC;
  endfor
endfunction
