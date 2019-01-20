function generatePOSCont(file, N, M)

## -*- texinfo -*-
## @deftypefn  {Function File} generatePOS (@var{file}, @var{N}, @var{M})
## Creates samples of POS (Partial observable system).
## The system has 3 possible state and for each state
## a singole continuos observable is generated with normal distribution
## with different mean value for each state and fixed variance
## 
## @var{file} is the file prefix
## 
## @var{N} is the number of samples to be created
##
## @var{M} is the length of samples
##
## @end deftypefn

  INITIALS = [0.8 0.1 0.1];

  TRANSITIONS = [
    0.1 0.8 0.1;
    0.1 0.1 0.8;
    0.1 0.1 0.8
  ];

  
  OBSERVABLES = [
    0.8 0.1 0.1;
    0.1 0.8 0.1;
    0.1 0.1 0.8
  ];

  OBSERVABLES_MEANS = [0.0 10.0 20.0];

  STD = 0.5;

  for i = 1 : N
    S = genSequence(M, INITIALS, TRANSITIONS);
    X = genObservables(S, OBSERVABLES);
    Y = genOutput(S, OBSERVABLES_MEANS, STD);
    csvwrite([file "_features_" int2str(i - 1) ".csv"], X(1 : end - 1 , : ));
    csvwrite([file "_labels_" int2str(i - 1) ".csv"], Y(2 : end, : ));  
  endfor

endfunction

function S = genState(P)
  [ n m ]= size(P);
  S = zeros(n, 1);
  for i = 1 : n
    S(i) = discrete_rnd([1 : m] - 1, P(i, :), 1, 1);
  endfor
endfunction

function Y = genSequence(N, INITIALS, TRANSITIONS)
  Y = zeros(N, 1);           
  Y(1) = genState(INITIALS);
  for i = 2 : N
    Y(i) = genState(TRANSITIONS(Y(i - 1) + 1, :));
  endfor
endfunction

function Y = genOutput(X, MEANS, STD)
  n = length(X);
  Y = randn(n, 1) * STD + MEANS'(X + 1);
endfunction

function Y = genObservables(X, P)
  m = length(P);
  n = length(X);
  Y = rand(n, m) <= P(X + 1, :);
endfunction
