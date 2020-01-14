function y = beta_multi(x)
% y = prod(gamma(x)) / gamma(sum(x));

x = x(:);
y_ln = sum(gammaln(x)) - gammaln(sum(x));
y = exp(y_ln);