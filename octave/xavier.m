function W = xavier(L)
  nl = length(L);
  assert(nl >= 2);
  W = cell(1, nl - 1);
  for i = 1 : nl - 1
    ni = L(i) + 1;
    no = L(i + 1);
    W{i} = randn(no, ni) * sqrt(2 / (no + ni));
  endfor
endfunction