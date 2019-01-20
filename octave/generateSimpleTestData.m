function generateSimpleTestData()
## -*- texinfo -*-
## @deftypefn  {Function File} generateTestData ()
## Creates samples of LSTM test case
## @end deftypefn
  NO_SAMPLES = 10
  MIN_LEN = 5
  MAX_LEN = 10
  MIN_TARGETS = 1
  MAX_TARGETS = 3
  FILE = "../src/test/resources/datatest";
  SEED = [1 2 3];
  printf "Generating test data ...\n";
  rand("state", SEED);
  generateSamples(NO_SAMPLES, MIN_LEN, MAX_LEN, MIN_TARGETS, MAX_TARGETS, FILE);
  printf "Completed.\n";
 
endfunction

function generateSamples(N, MIN_LEN, MAX_LEN, MIN_TARGETS, MAX_TARGETS, file)
  for i = 1 : N
    len = randi(MAX_LEN - MIN_LEN + 1) + MIN_LEN - 1;
    maxTargets = min(len, MAX_TARGETS);
    minTargets = min(maxTargets, MIN_TARGETS);
    noTargets = randi(maxTargets - minTargets + 1) + minTargets - 1;
    [X Y] = generateSequence(len, noTargets);
    csvwrite([file "/features_" int2str(i - 1) ".csv"], X);
    csvwrite([file "/labels_" int2str(i - 1) ".csv"], Y);
  endfor
endfunction

## Generate a sequence with N length and M positive matches
function [X Y] = generateSequence(N, M)
  X = rand(N, 1);
  Y = -ones(N, 1);
  TARGETS = [1 (randperm(N-1)(1 : M)+1)];
  X(TARGETS, 1) = X(1, 1);
  Y(TARGETS, 1) = 1;
endfunction
  