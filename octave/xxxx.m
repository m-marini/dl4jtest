N = 10;
for i = 1 :N
  len = randi(5) + 4;
  csvwrite(["../data/feature_" int2str(i-1) ".csv"], randn(len,3));
  csvwrite(["../data/label_" int2str(i-1) ".csv"], idx_rnd(len,3));  
endfor
