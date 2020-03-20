function y_hyp = learn_dir_01(x,D,Y,X,alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0-1 Classifier, Dirichlet Prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (numel(Y) ~= size(alpha,1)) || (numel(X) ~= size(alpha,2))
    error('Set and Alpha sizes do not match.');
end

for idx_x = 1:numel(X)
    if X{idx_x} == x
        break
    end
end

N_bar_x = zeros(size(Y));
for m = 1:numel(Y)
    for n = 1:numel(D)
        N_bar_x(m) = N_bar_x(m) + (Y{m} == D(n).y)*(x == D(n).x);
    end
end

[~,idx] = max(alpha(:,idx_x) + N_bar_x);
y_hyp = Y{idx};