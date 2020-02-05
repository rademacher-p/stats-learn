function y_est = learn_dir_SE(x,D,Y,X,alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Squared-Error Estimator, Dirichlet Prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (numel(Y) ~= size(alpha,1)) || (numel(X) ~= size(alpha,2))
    error('Set and Alpha sizes do not match.');
end

for m_x = 1:numel(X)
    if x == X{m_x}
        idx_x = m_x;
    end
end

alpha_x = sum(alpha(:,idx_x));

N_x = 0;
sum_y = 0;
for n = 1:numel(D)
    if D(n).x == x
        N_x = N_x + 1;
        sum_y = sum_y + D(n).y;
    end
end


y_est = (cell2mat(Y)'*alpha(:,idx_x) + sum_y) / (alpha_x + N_x);