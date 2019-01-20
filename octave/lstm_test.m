function Y = lstm_test(X, WG, WS, WR, WF)

## -*- texinfo -*-
## @deftypefn  {Function File} {@var{Y}} = lstm_test(@var{X}, @var{WG}, @var{WS}, @var{WR}, @var{WF})
## Compute the outputs of a LSTM neural network with samples
##
## @var{X} is a matrix (t,m) with t steps of m input siganls
##
## @var{WG} is a matrix containing the weights of input cells (n, m + 1)
##
## @var{WS} is a matrix containing the weights of input gate (n, m + 1)
##
## @var{WR} is a matrix containing the weights of forget gate (n, m + 1)
##
## @var{WF} is a matrix containing the weights of output gate (n, m + 1)
##
## @var{WF} is a matrix containing the weights of output gate (n, m + 1)
##
## The return values is
##
## @var{Y} the matrix (t, n) with t steps of n output siganls
##
## @end deftypefn
  [t,m] = size(X);
  n = size(WG, 1);
  Y = zeros(t, n);
  Z = zeros(7, n);
  for i = 1 : m
    Z = lstm_forward(X(i,:), Z(4,:), WG, WS, WR, WF);
    Y(i,:) = Z(1,:);
  endfor
endfunction