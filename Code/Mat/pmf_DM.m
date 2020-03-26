function y = pmf_DM(x,alpha_0,alpha)
% Dirichlet-Multinomial PMF evaluation
% Input: x (sample length by number of samples

[L_s,N_s] = size(x);

N = unique(sum(x,1));
if numel(N) > 1
    error('Samples should have the same sum.');
end

y = zeros(1,N_s);
for ii = 1:N_s
    y(ii) = coef_multi(x(:,ii)) * beta_multi(alpha_0*alpha + x(:,ii)) / beta_multi(alpha_0*alpha);
end
