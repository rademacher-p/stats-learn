function y_hyp = learn_dir_01(Y,X,alpha,D,x)

% [~,idx_x] = ismember(x,X);
for m_x = 1:numel(X)
    if x == X{m_x}
        idx_x = m_x;
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