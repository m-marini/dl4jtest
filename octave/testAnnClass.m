clear all;
N = 10000;
NI = 300;
L = [1 2];

function [ X Y ] = createTest(N)
  Y = zeros(N, 2);
  X = zeros(N, 1);
  I = randi(2, N, 1);
  I1 = find(I == 1);
  I2 = find(I == 2);
  Y(I1, 1) = 1;
  Y(I2, 2) = 1;
  X(I1) = randn(length(I1), 1) - 1;
  X(I2) = randn(length(I2), 1) + 1;
endfunction

function [ W ERR ] = train(X, Y, L, NI)
  ERR = zeros(NI, 1);
  W = xavier(L);
  ETA = 0.3;
  for i = 1 : NI
    H = forward(W, X);
    [W E] = backward(W, Y, H, ETA, 0, 0, @sign);
    ETA *= 0.99;
    ERR(i) = E;
    printf("Error %g\r", E);
    fflush(1);
  endfor
  printf("\n");
endfunction

## Create test
[ X Y ] = createTest(N);

## Normalize
X = (X - mean(X)) / std(X);

[W ERR] = train(X, Y, L, NI);

plot(ERR);
grid minor on;
