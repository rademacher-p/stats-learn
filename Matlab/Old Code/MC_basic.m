%%%%%%%%%% Simulation: Basic Model

clear;

%%% Inputs

% fcn_loss = @loss_SE;
% fcn_learn = @learn_opt_SE_basic;
% Y = num2cell((1:4)'/4);             % output set

fcn_loss = @loss_01;
fcn_learn = @learn_opt_01_basic;
Y = {'a';'b';'c';'d'};
% Y = num2cell((1:4)');


alpha = ones(4,1);         % prior PDF parameters
N = 1;                     % Number of training data

N_mc = 20000;                  % Number of monte carlo iterations




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
    
    if mod(idx_mc,1000) == 0
        fprintf('Monte Carlo iteration %i/%i \n',idx_mc,N_mc);
    end
        
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
Risk_e = mean(loss)



if strcmpi(func2str(fcn_learn),'learn_opt_01_basic') && (sum(alpha==1) == M)
    temp = -1/(N+M);
    for m = 1:M
        for n = ceil((N+M)/m):(N+M)
            temp = temp + -1/(N+M) * (-1)^m * nchoosek(M,m) * prod(1 - m*n./(N+(1:M-1)));
        end
    end 
    Risk_a = temp
end

if strcmpi(func2str(fcn_learn),'learn_opt_SE_basic')
    alpha_0 = sum(alpha);
    mu_theta = alpha / alpha_0;
    mu_y_prior = cell2mat(Y)'*mu_theta;

    var_y = (cell2mat(Y).^2)'*mu_theta - (cell2mat(Y)'*mu_theta).^2;
    Risk_a = alpha_0/(alpha_0+1)*(alpha_0+N+1)/(alpha_0+N) * var_y
end









