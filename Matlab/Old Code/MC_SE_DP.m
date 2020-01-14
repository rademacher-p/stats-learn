%%%%%%%%%% Simulation: Basic Model, Squared-Error Loss

clear;

%%% Inputs
Y = (1:5)'/5;             % output set
alpha = ones(5,1);         % prior PDF parameters

N = 50;                     % Number of training data

N_mc = 10000;                  % Number of monte carlo iterations



%%% Generate PMF set
if numel(Y) == numel(alpha)
    M = numel(Y);
else
    disp('Error: Set and Alpha sizes do not match.');
    return
end

alpha_0 = sum(alpha);
mu_theta = alpha / alpha_0;
mu_y_prior = Y'*mu_theta;


%%% Iterate
loss = zeros(1,N_mc);

for idx_mc = 1:N_mc
     
    %%% Generate training data
    D = zeros(N,1);
    
    P_c = mu_theta;
    
    temp = find(rand <= cumsum(P_c));
    idx_d = temp(1);
    
    D(1) = Y(idx_d);
    
    
    for n = 2:N
        P_c = P_c*(alpha_0+n-2);
        P_c(idx_d) = P_c(idx_d)+1;
        P_c = P_c / (alpha_0 + n-1);
        
        temp = find(rand <= cumsum(P_c));
        idx_d = temp(1);

        D(n) = Y(idx_d);
    end
   

    %%% Generate test datum
    P_c = P_c*(alpha_0+N-1);
    P_c(idx_d) = P_c(idx_d)+1;
    P_c = P_c / (alpha_0 + N);

    temp = find(rand <= cumsum(P_c));
    idx_d = temp(1);

    y = Y(idx_d);
       


    %%% Optimal Estimator
    y_est = (alpha_0/(alpha_0+N))*mu_y_prior + (N/(alpha_0+N))*mean(D);
       

    %%% Assess loss    
    loss(idx_mc) = (y_est - y)^2;

end

Risk = mean(loss)

var_y = (Y.^2)'*mu_theta - (Y'*mu_theta).^2;
Risk_a = alpha_0/(alpha_0+1)*(alpha_0+N+1)/(alpha_0+N) * var_y





