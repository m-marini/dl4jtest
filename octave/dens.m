function Y = dens(X, P, S)
  Y = countDens(X, P, S) / (S .^ 2);
endfunction