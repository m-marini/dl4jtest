clear all;
N = 10000;
L = [3 2 2];
X = randn(N, 3);
Y = [zeros(N, 1) ones(N, 1)];
W = xavier(L);
for i = 1 : 100
  H = forward(W, X);
  [W E] = backward(W, Y, H);
  printf("Error %g\r", E);
  fflush(1);
endfor
printf("\n");
forecast(W, X);
