function Y = lstm_forward(x, cec, WG, WS, WR, WF)

## -*- texinfo -*-
## @deftypefn  {Function File} {@var{Y} = lstm_forward (@var{x}, @var{cec}, @var{WG}, @var{WS}, @var{WR}, @var{WF})
## Compute the status of LSTM neural network.
## 
## @var{x} is a vector containing the input signals (1, m) 
##
## @var{cec} is a vector containing the status values of CECs (1, n)
##
## @var{WG} is a matrix containing the weights of input cells (n, m + 1)
##
## @var{WS} is a matrix containing the weights of input gate (n, m + 1)
##
## @var{WR} is a matrix containing the weights of forget gate (n, m + 1)
##
## @var{WF} is a matrix containing the weights of output gate (n, m + 1)
##
## The return value @var{Y} is a matrix containing the status of network (6, n)
## @itemize
## @item
## Y(1,:) = output values
## @item
## Y(2,:) = output gates (YF)
## @item
## Y(3,:) = output cells (YH)
## @item
## Y(4,:) = status values of CECs
## @item
## Y(5,:) = forget gates (YR)
## @item
## Y(6,:) = input gates (YS)
## @item
## Y(7,:) = input cells (YG)
## @end itemize
##
## @end deftypefn

  x1 = [1 x]';
  n = size(WG, 1);
  Y = zeros(7, n);
  Y(7, :) = tanh(WG * x1);
  Y(6, :) = tanh(WS * x1);
  Y(5, :) = tanh(WR * x1);
  Y(4, :) = cec .* Y(5, :) + Y(6, :) .* Y(7, :);
  Y(3, :) = tanh(WF * x1);
  Y(2, :) = Y(4, :);
  Y(1, :) = Y(2, :) .* Y(3, :);
endfunction