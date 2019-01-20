function Y = idx_rnd(N, M)
  Y = zeros(N,M);
  for i= 1:N
    Y(i,randi(M))=1;
  endfor
endfunction
