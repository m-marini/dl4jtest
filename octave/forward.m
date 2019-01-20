function H=forward(W, X, type="softmax")
  [n ni] = size(X);
  nl = length(W) + 1;
  no = size(W{end}, 1);
  H = cell(nl, 1);

  IN = [ones(n,1) X];
  H{1}.out = X;

  for i = 2 : nl - 1
     Z = IN * W{i - 1}';
     H{i}.out = tanh(Z);
     H{i}.grad = @tanhError;
     IN = [ones(n,1) Z];
  endfor

  Z = IN * W{end}';

  switch type
  case "linear"
    H{end}.out = Z;
    H{end}.grad = @identError;
  case "logistic"
    H{end}.out = logistic_cdf(Z);
    H{end}.grad = @logisticError;
  case "softmax"
    Z = exp(Z);
    H{end}.out = Z ./ sum(Z, 2);
    H{end}.grad = @softmaxError;
  endswitch
endfunction

function DY1 = tanhError(Y, DY);
  DY1 = DY .* (1 + Y) .* (1 - Y);
endfunction

function DY1 = logisticError(Y, DY);
  DY1 = DY .* Y .* (1 - Y);
endfunction

function DY = identError(Y, DY);
  DY1 = DY;
endfunction

function DY1 = softmaxError(Y, DY);
  DY1 = Y .* DY;
  DY2 = sum(DY1 .* Y, 2);
  DY1 = DY1 - DY2;
endfunction
