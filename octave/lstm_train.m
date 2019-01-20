function [WG1 WS1 WR1 WF1 E] = lstm_train(ITER, DATA, WG, WS, WR, WF, eta, deta)

## -*- texinfo -*-
## @deftypefn  {Function File} {[@var{WG1} @var{WS1} @var{WR1} @var{WF1} @var{E} ]} = lstm_train(@var{ITER}, @var{DATA}, @var{WG}, @var{WS}, @var{WR}, @var{WF}, @var{eta}, @var{deta})
## Train a LSTM neural network with samples
## 
## @var{ITER} is the number of iterations
##
## @var{DATA} is a cell array containing the samples (n,2)
## @itemize
## @item
## Y@{ : , 1 @} = matrix (nt, no) of nt step of no expected values
## @item
## Y@{ : , 2 @} = <matrix (nt, no) of nt step of ni expected values
## @end itemize
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
## @var{ETA} is learning rate
##
## @var{DETA} is learning rate decay step factor
##
## The return values are
##
## @var{WG1} is the resulting matrix with the weights of input cells (n, m + 1)
##
## @var{WS1} is the resulting matrix with the weights of input gate (n, m + 1)
##
## @var{WR1} is the resulting matrix with the weights of forget gate (n, m + 1)
##
## @var{WF1} is the resulting matrix with the weights of output gate (n, m + 1)
##
## @var{E} is a matrix with square mean errors by step,
##
## @end deftypefn

  m = size(DATA,1);
  WG1 = WG;
  WS1 = WS;
  WR1 = WR;
  WF1 = WF;
  E = zeros(ITER, 1);
  for i = 1 : ITER
    E(i) = 0;
    for j = 1 : m
      [ER WG1 WS1 WR1 WF1] = lstm_train_ep(DATA{j, 1}, DATA{j, 2}, WG1, WS1, WR1, WF1, eta);
      E(i) = E(i) + ER;
     endfor
     eta = eta * deta;
  endfor
endfunction

function [E WG1 WS1 WR1 WF1] = lstm_train_ep(X, Y, WG, WS, WR, WF, eta)
  [t m] = size(X);
  n = size(Y, 2);
  E = 0;
  WG1 = WG;
  WS1 = WS;
  WR1 = WR;
  WF1 = WF;
  YY = zeros(7, n);
  for i = 1 : t
    Y1 = lstm_forward(X(i, :), YY(4, :), WG1, WS1, WR1, WF1);
    ERR = Y(i,:) - Y1(1,:);
    [WG1 WS1 WR1 WF1] = lstm_backward(ERR, X(i, :), Y1, YY(4, :), WG1, WS1, WR1, WF1, eta);
    Y1 = YY;
    E = E + sum(ERR .^2);
  endfor
endfunction

function [WG1 WS1 WR1 WF1] = lstm_backward(E, X, Y, SCM, WG, WS, WR, WF, eta)
  X1 = [1 X];
  # DYF = YH (1 - YF)^2 I
  DYF = tensprod(Y(3,:) .* (1 - Y(2,:)).^2, X1);
  # DYR = YF (1 - YH)^2 SCM (1 - YR)^2 I
  DSCM = Y(2,:) .* (1 - Y(3,:)).^2;
  DYR = tensprod(DSCM .* SCM .* (1 - Y(5,:)).^2, X1);
  # DYS = YF (1 - YH)^2 YG (1 - YS)^2 I
  DYS = tensprod(DSCM .* Y(7,:) .* (1 - Y(6,:)).^2, X1);
  # DYG = YF (1 - YH)^2 YS (1 - YG)^2 I
  DYG = tensprod(DSCM .* Y(6,:) .* (1 - Y(7,:)).^2, X1);
  
  WF1 = WF - eta * DYF .* E';
  WR1 = WR - eta * DYR .* E';
  WS1 = WS - eta * DYS .* E';
  WG1 = WG - eta * DYG .* E';
endfunction

#
# Y(i,j) = A(i) * B(j)
#
function Y = tensprod(A,B)
  n = size(A,2);
  m = size(B,2);
  Y = zeros(n,m);
  Y = repmat(A',1,m) .* repmat(B,n,1);
endfunction