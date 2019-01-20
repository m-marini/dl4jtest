Y=[];
for i = 1 : 3000
  Y = [ Y;
  csvread(["../data/pos1train/model_labels_" num2str(i-1) ".csv"])
  ];
endfor