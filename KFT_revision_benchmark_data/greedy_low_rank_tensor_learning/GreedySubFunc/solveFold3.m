function [delta, Sol] = solveFold3(Y, X, Sol)
% The goal of the following functions is to (1) Find the optimal rank-1
% direction in the given mode and (2) Report the amount of decrease in the
% objective function
[q, p, r] = size(Sol);
Max_Iter = 100;
obj = zeros(Max_Iter, 1);
A = ones(q, p);
step = 1e-3;

for i = 1:Max_Iter
    [obj(i), G] = findGrad3(Y, X, A);
    A = A + step * G;
end

u = zeros(r, 1);
for i = 1:r
    u(i) = trace(A*X{i}*Y{i}')/norm(A*X{i}, 'fro')^2;
end

SS = u*reshape(A', 1, p*q); %%%
Sol = fld(SS, 3, q);

% Computing delta
delta = 0;
for ll = 1:r
    delta = delta + norm(Y{ll}, 'fro')^2 - norm(Y{ll} - squeeze(Sol(:, :, ll))*X{ll}, 'fro')^2;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [obj, G] = findGrad3( Y, X, A )
obj = 0;
G = 0*A;
for i = 1:length(Y)
    m = norm(A*X{i}, 'fro')^2;
    tr = trace(A*X{i}*Y{i}');
    obj = obj + tr^2/m;
    mp = 2*tr*(Y{i}*X{i}');
    G = G + mp/m - 2*A*(X{i}*X{i}')*(tr^2)/(m^2);
end
end