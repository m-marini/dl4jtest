function generateTestData()
## -*- texinfo -*-
## @deftypefn  {Function File} generateTestData ()
## Creates samples of LSTM test case
## @end deftypefn

  SEED = [1 2 3];
  printf "Generating test data ...\n";
  rand("state", SEED);
  generateSamples(10, 5, 10, "../src/test/resources/datatest");
  printf "Completed.\n";
 
endfunction

function generateSamples(N, MIN_LEN, MAX_LEN, file)
  for i = 1 : N
    len = randi(MAX_LEN - MIN_LEN + 1) + MIN_LEN - 1;
    [X Y] = generateSequence(len);
    csvwrite([file "/features_" int2str(i - 1) ".csv"], X);
    csvwrite([file "/labels_" int2str(i - 1) ".csv"], Y);
  endfor
endfunction

function [X Y] = generateSequence(N)
  X = rand(N, 3);
  Y = zeros(N, 3);
  S = zeros(3, 3);
  j = 1;
  for i = 1 : N
    Y(i, 3) = S(j, 3);
    Y(i, 2) = S(mod(j, 3) + 1, 2);
    Y(i, 1) = S(mod(j + 1, 3) + 1, 1);
    S(j, :) = X(i, :);
    j = mod(j, 3) + 1;
  endfor
endfunction
  