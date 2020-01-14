%%%%%%%%%%

clear;

M = 10;

Y = (1:M)'/M;
E_theta = ones(M,1)/M;

N_p = (1:10)';
alpha_0_p = (0.5:0.5:10)'; 


if sum(E_theta < 0) ~= 0
    disp('Error: Invalid prior mean')
    return
elseif sum(E_theta) ~= 1
    disp('Warning: Prior mean not normalized');
    E_theta = E_theta / sum(E_theta);
end


var_y = (Y.^2)'*E_theta - (Y'*E_theta).^2;

R_p = zeros(numel(N_p),numel(alpha_0_p));
for idx_n = 1:numel(N_p)
    for idx_a = 1:numel(alpha_0_p)        
        N = N_p(idx_n);
        alpha_0 = alpha_0_p(idx_a);
        
        R_p(idx_n,idx_a) = alpha_0/(alpha_0+1)*(alpha_0+N+1)/(alpha_0+N) * var_y;
    end
end


figure(1); clf;
plot(N_p,R_p); grid on;
xlabel('N'); ylabel('Risk');

figure(2); clf;
plot(alpha_0_p,R_p'); grid on;
xlabel('alpha_0'); ylabel('Risk');

