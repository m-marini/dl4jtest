function [X Y] = createTest(DIST, N)

## -*- texinfo -*-
## @deftypefn  {Function File} {[@var{X} @var{Y}] = createTest (@var{DIST}, @var{N})
## Create test sample.
## 
## @var{DIST} is a matrix containing the preference, mean values and standard deviation for samples by labels
## 
## @var{N} is the number of samples to be created
##
## The return value @var{X} is a vector of values and @var{Y} is a vector of the label indexes
##
## @end deftypefn
  M = size(DIST, 1);
  
  X = zeros(N, 1);
  Y = zeros(N, 1);
  
  Y = randpref(DIST(:,1),N);
  X = randn(N, 1);
  for i = 1 : M
    X(Y==i) = X(Y==i) * DIST(i,3) + DIST(i,2);
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
