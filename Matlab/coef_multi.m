function y = coef_multi(x)
% y = factorial(sum(x)) / prod(factorial(x));

x = x(:);
sx = sort(x);

nums = (x(end)+1):sum(x);
x(end) = [];

dens = [];
for idx = 1:numel(x)
    dens = [dens, 1:x(idx)];
end
dens = sort(dens);

y = prod(nums./dens);
