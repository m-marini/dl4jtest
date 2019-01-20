P = [0 : 0.01 : 1]';
n = length(P);

[P Q] = meshgrid(0:0.05:1,0:0.05:1);
A = 0;
B = 3;
C = 1;
D = 0;
R = A;
S = C;
T = B;
U = D;
F = A .* P .* Q + B .* P .* (1 - Q) + C .* (1 - P) .* Q + D .* (1 - P) .* (1 - Q);
G = R .* P .* Q + S .* P .* (1 - Q) + T .* (1 - P) .* Q + U .* (1 - P) .* (1 - Q);
surfc(P, Q, F);

aaaaa;

m = 10000;
F = zeros(n, 1);
G = zeros(n, 1);
for i = 1 : n
  E1 = binornd(1, 0.75, m, 1);
  E2 = binornd(1, P(i), m, 1);
  F(i) = mean(3 * (E1 & !E2) + (!E1 & E2));
  G(i) = mean((E1 & !E2) + 3 * (!E1 & E2));
endfor
plot(P , [F G]);
grid minor on;
legend("F", "G");
xlabel("Q(A)");
aaaaa;

P = [0:0.05:1]';
Q = P;
A = 0;
B = 3;
C = 1;
D = 0;
R = A;
S = C;
T = B;
U = D;
F = A .* P .* Q + B .* P .* (1 - Q) + C .* (1 - P) .* Q + D .* (1 - P) .* (1 - Q);
G = R .* P .* Q + S .* P .* (1 - Q) + T .* (1 - P) .* Q + U .* (1 - P) .* (1 - Q);
plot(P,[F G]);
grid minor on;
legend("F", "G");

aaaaa;

[P Q] = meshgrid(0:0.05:1,0:0.05:1);
A = 0;
B = 3;
C = 1;
D = 0;
R = A;
S = C;
T = B;
U = D;
F = A .* P .* Q + B .* P .* (1 - Q) + C .* (1 - P) .* Q + D .* (1 - P) .* (1 - Q);
G = R .* P .* Q + S .* P .* (1 - Q) + T .* (1 - P) .* Q + U .* (1 - P) .* (1 - Q);
surfc(P, Q, F-G);
