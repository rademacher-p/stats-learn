%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conditional and Min. Bayes Risk, Dirichlet Learner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

%%% Inputs

N_mc = 50000;                  % Number of monte carlo iterations

N = 10;                     % Number of training data


Y = num2cell((1:4)'/4);             % output set
X = num2cell((1:3)');             % input set
% Y = {'a';'b';'c';'d'};
% X = num2cell((1:3)');


% alpha = ones(4,3);         % prior PDF parameters
alpha = 10*rand(4,3);
fcn_prior = @(N_mc)dirrnd(alpha,N_mc);
% theta_0 = dirrnd(alpha,1);
% fcn_prior = @(N_mc)repmat(theta_0,[1,1,N_mc]);


% fcn_loss = @loss_SE;
% fcn_learn = @(x,D)learn_dir_SE(x,D,Y,X,alpha);
% fcn_risk_a = @(Y,X,N)risk_min_dir_SE(Y,X,N,alpha);
% % fcn_risk_a = @(Y,X,N)risk_cond_dir_SE(Y,X,N,theta_0,alpha);


fcn_loss = @loss_01;
fcn_learn = @(x,D)learn_dir_01(x,D,Y,X,alpha);
fcn_risk_a = @(Y,X,N)risk_min_dir_01(Y,X,N,alpha);
% fcn_risk_a = @(Y,X,N)risk_cond_dir_01(Y,X,N,theta_0,alpha);





%%% Generate Models
theta_mc = fcn_prior(N_mc);


%%% Iterate
M_y = numel(Y);
M_x = numel(X);

loss = zeros(N_mc,1);
for idx_mc = 1:N_mc
    
    if mod(idx_mc,1000) == 0
        fprintf('Monte Carlo iteration %i/%i \n',idx_mc,N_mc);
    end
    
            
    theta = theta_mc(:,:,idx_mc);    
    theta_cs = cumsum(theta(:));
        
    
    %%% Training Data       
    D = struct('y',cell(N,1),'x',cell(N,1));
    for n = 1:N
        temp = find(rand <= theta_cs);
        [idx_d_y,idx_d_x] = ind2sub([M_y,M_x],temp(1)); 
        
        D(n).y = Y{idx_d_y};
        D(n).x = X{idx_d_x};
    end
    
    
    
    %%%% change to singular structure for speed??
    
    %%%% use multi rand instead???
    
%     temp = mnrnd(N,theta(:))';
%     n_bar = reshape(temp,[M_y,M_x]);
%     nb_cs = [0; cumsum(temp)];
%     idx_cs = [nb_cs(1:end-1)+1,nb_cs(2:end)];
    
%     D = struct('y',cell(N,1),'x',cell(N,1));
%     for ii_x = 1:M_x
%         for ii_y = 1:M_y
%             idx = sub2ind([M_y,M_x],ii_y,ii_x);
%             nn = idx_cs(idx,:);
%             for n = idx_cs(idx,1):idx_cs(idx,2)
%                 D(n) = struct('y',Y(ii_y),'x',X(ii_x));
%             end
%         end
%     end
    
    
    
    
    
    %%% Generate test datum
    temp = find(rand <= theta_cs);
    [idx_d_y,idx_d_x] = ind2sub([M_y,M_x],temp(1)); 
    
    y = Y{idx_d_y};
    x = X{idx_d_x};
       

    %%% Create Decision
    h = fcn_learn(x,D);

    
    %%% Assess Loss
    loss(idx_mc) = fcn_loss(h,y);

end

%%% Empirical Risk
Risk_e = mean(loss);


%%% Analytical Risk
Risk_a = fcn_risk_a(Y,X,N);


%%%%% Results
fprintf('\n Bayes Risk (Empirical) = %f \n Bayes Risk (Analytical) = %f \n',Risk_e,Risk_a);












