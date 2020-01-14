function y_hyp = learn_dir_01_basic(Y,alpha,D)

N_bar = zeros(size(Y));
for m = 1:numel(Y)
    for n = 1:numel(D)
        N_bar(m) = N_bar(m) + (Y{m} == D{n});
    end
end

[~,idx] = max(alpha + N_bar);
y_hyp = Y{idx};