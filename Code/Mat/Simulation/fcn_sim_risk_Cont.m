function Risk_e = fcn_sim_risk_Cont(N_mc,Y,X,N,fcn_prior,fcn_learn,fcn_loss)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation: Supervised Learning and Random Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Generate Models
theta_mc = fcn_prior(N_mc);


%%% Iterate
M_y = numel(Y);
M_x = numel(X);
    
loss = zeros(N_mc,1);
for idx_mc = 1:N_mc
        
    theta = theta_mc(:,:,idx_mc);
    theta_cs = cumsum(theta(:));
    
    
    thetam_icdf = @(p)icdf(makedist('Beta','a',5,'b',5),p);
    thetac_icdf = @(x,p)icdf(makedist('Beta','a',10*x,'b',10*(1-x)),p);

    %%% Training Data       
    D = struct('y',cell(N,1),'x',cell(N,1));
    for n = 1:N
        D(n).x = thetam_icdf(rand);
        D(n).y = thetac_icdf(D(n).x, rand);
        
%         temp = find(rand <= theta_cs);
%         [idx_d_y,idx_d_x] = ind2sub([M_y,M_x],temp(1)); 
%         
%         D(n).y = Y{idx_d_y};
%         D(n).x = X{idx_d_x};
    end
    
    
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








