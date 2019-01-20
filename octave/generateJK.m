function [X Y] = generateJK(file, N, MIN_SIZE, MAX_SIZE)
## -*- texinfo -*-
## @deftypefn  {Function File} generateJK (@var{N})
## Creates a sample sequence of JK Flip Flop.
## 
## @var{N} is the sequence length of samples
##
##
## The return values @var{X} is a matrix containing the input values and
## @var{Y} is a matrix containing the expected output values
## @end deftypefn

  for i = 1 : N
    [X Y] = generateJKSequence(randi(MAX_SIZE - MIN_SIZE + 1) + MIN_SIZE - 1);
    csvwrite([file "features_" int2str(i - 1) ".csv"], X);
    csvwrite([file "labels_" int2str(i - 1) ".csv"], Y);
  endfor

endfunction

function [X Y] = generateJKSequence(N)
## -*- texinfo -*-
## @deftypefn  {Function File} generateJK (@var{N})
## Creates a sample sequence of JK Flip Flop.
## 
## @var{N} is the sequence length of samples
##
##
## The return values @var{X} is a matrix containing the input values and
## @var{Y} is a matrix containing the expected output values
## @end deftypefn

  X = randi(2, N, 2) - 1;
  Y = zeros(N, 1);

  Y0 = 0;
  for i = 1 : N
    J = X(i, 1);
    K = X(i, 2);
    if !J & K
      Y0 = 0;
    elseif J & !K
      Y0 = 1;
    elseif J & K
      Y0 = !Y0;
    endif
    Y(i) = Y0;
  endfor
endfunction