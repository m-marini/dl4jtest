clear all;

function main
  F = cell(2, 1);

  F{1}.p = 0.95;
  F{1}.m = [-0.5 -0.5];
  F{1}.s = eye(2) * sqrt(0.9);

  F{2}.p = 0.05;
  F{2}.m = [0.5 0.5];
  F{2}.s = eye(2) * sqrt(0.2);

  P = [64 64];
  C = [-1 -1; 1 1];
  NSAMPLE = 100000;
  NITER = 300;
  LAYERS = [2 10 2];
  ETA = 0.3;

  ## Create samples
[ SAMPLES EXP ]= createTestSamples(NSAMPLE, F);
  
  #scatter3(SAMPLES(:,1), SAMPLES(:,2), EXP, 1, EXP+1);
  EXP = vectorizeIndex(EXP, 2);

  ## Train model
  [W XM XS] = train(LAYERS, SAMPLES, EXP, NITER, ETA);
  
  [X Y Z] = resultChart(W, P, C, XM, XS);

  surf(X, Y, Z);
  #contourf(X, Y, Z);
  grid minor on;
endfunction

function [Y MEAN STD] = computeNormalization(X)
  MEAN = mean(X);
  STD = std(X);
  Y = normalize(X, MEAN, STD);
endfunction

function Y = normalize(X, MEAN, STD)
  Y = (X - MEAN) ./ STD;
endfunction

function Y = vectorizeIndex(X, M)
  N = size(X, 1);
  Y = zeros(N, M);
  for i = 1 : N
    Y(i, X(i)) = 1;
  endfor
endfunction

function [W XM XS] = train(LAYERS, SAMPLES, EXP, NITER, ETA)
  ## Normalize samples
  [SAMPLES XM XS] = computeNormalization(SAMPLES);

  W = xavier(LAYERS);
  E0 = 1e18;
  E = E0;
  for i = 1 : NITER
    H = forward(W, SAMPLES, "softmax");
    [W1 E1] = backward(W, EXP, H, ETA, 0, 0);
    printf("\r%d %g %g", i, E, E1);
    fflush(1);
    ## E1 * 1000 > (1000 - 1 )* E
    if E != E0 && E1 * 1000 >= 999 * E 
      break
    else
      W = W1;
      E = E1;
    endif
  endfor
  printf("\n");
endfunction

function [X Y Z] = resultChart(W, P, C, XM, XS)
  [ X Y ] = meshgrid(
    (0 : P) * (C(2, 1) - C(1, 1)) / P(1) + C(1, 1),
    (0 : P) * (C(2, 2) - C(1, 2)) / P(2) + C(1, 2));
  SAMPLES = normalize([X(:) Y(:)], XM, XS);
  OUT = forecast(W, SAMPLES);
  GR = double(OUT(:, 2) - OUT(:, 1));
  Z = reshape(GR, P(1) + 1, P(2) + 1);
endfunction

main();
