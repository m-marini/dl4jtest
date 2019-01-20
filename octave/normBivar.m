function X = normBivar(N, MEAN, S) 
  ## Create samples
  R = randn(N, 2);
  X = zeros(N, 2);
  X = R * S + MEAN;
endfunction
