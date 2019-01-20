function [W1 E] = backward(W, Y, H, ETA=0.3, L1=0, L2=0, gradFunc=@(X)X)
  n = size(H{1}.out, 1);
  nl =length(H);
  no = size(H{nl}.out, 2);
  DELTA = (Y - H{end}.out);
    
  E = sum(mean(DELTA .^ 2));
  for i = 1 : nl - 1
    WW = W{i}(:, 2 : end)(:);
    E += L1 * sum(abs(WW)) + L2 / 2 * sum(WW .^ 2);
  endfor

  W1 = W;

  for i = nl - 1 : -1 : 1
    X = H{i}.out;
    Y = H{i + 1}.out;
    DELTA = H{i + 1}.grad(Y, DELTA);
    [GRAD DELTA] = gradZW(W{i}, X, DELTA, ETA, L1, L2);
    W1{i} += ETA * gradFunc(GRAD);
  endfor
endfunction

function [GRAD DELTAO] = gradZW(W, X, DELTA, ETA, L1, L2)
  n = size(X, 1);
  GRAD = DELTA' * [ones(n, 1) X] / n;
  GRAD(: , 2 : end) += L1 * sign(W(:, 2 : end)) + L2 * W(:, 2 : end);
  DELTAO = DELTA * W(: , 2: end);
endfunction