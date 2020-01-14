function y_est = learn_dir_SE_basic(Y,alpha,D)

N = numel(D);
alpha_0 = sum(alpha);

y_est = (cell2mat(Y)'*alpha + sum(cell2mat(D))) / (alpha_0 + N);
