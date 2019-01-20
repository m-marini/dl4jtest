function Y = countDens(X, P, S)
  D2 = zeros(size(X, 1), 1);
  D2 = sum((X - P) .^ 2, 2);
  Y = sum(D2 < S .^ 2);
endfunction