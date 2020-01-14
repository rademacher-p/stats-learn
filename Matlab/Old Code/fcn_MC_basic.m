function Risk_e = fcn_MC_basic(Y,alpha,N,fcn_loss,fcn_learn,N_mc)
%%%%%%%%%% Simulation: Basic Model

%%% Generate Models
if numel(Y) == numel(alpha)
    M = numel(Y);
else
    disp('Error: Set and Alpha sizes do not match.');
    return
end

temp = zeros(M,N_mc);
for m = 1:M
    temp(m,:) = gamrnd(alpha(m),1,[1,N_mc]);
end
theta_mc = temp ./ (ones(M,1)*sum(temp,1)); 



%%% Iterate
loss = zeros(1,N_mc);
for idx_mc = 1:N_mc
    
%     if mod(idx_mc,1000) == 0
%         fprintf('Monte Carlo iteration %i/%i \n',idx_mc,N_mc);
%     end
        
    %%% Training Data
    D = cell(N,1);
    for n = 1:N
        temp = find(rand <= cumsum(theta_mc(:,idx_mc)));
        idx_d = temp(1);
        
        D{n} = Y{idx_d};
    end
    
    %%% Generate test datum
    temp = find(rand <= cumsum(theta_mc(:,idx_mc)));
    idx_d = temp(1);
    
    y = Y{idx_d};   
   

    %%% Create Hypothesis
    y_est = fcn_learn(Y,alpha,D);
    
    %%% Assess Loss
    loss(idx_mc) = fcn_loss(y_est,y);

end

%%% Empirical Risk
Risk_e = mean(loss);











