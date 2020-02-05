function Risk_a = risk_cond_dir_01(Y,X,N,theta,alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conditional Risk, 0-1 Loss, Dirichlet Learner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Risk_a = 1;

for idx_x = 1:numel(X)
    
    theta_m = sum(theta(:,idx_x));
    theta_c = theta(:,idx_x) ./ theta_m;
    
    temp_m = 0;
    for n_m = 0:N
        
        N_bar = N_bar_set_gen([numel(Y),1],n_m);        
        P_N_c = mnpdf(permute(N_bar,[3,1,2]),theta_c');
        
        temp_c = 0;
        for idx_c = 1:size(N_bar,3)
            [~,idx_max] = max(alpha(:,idx_x) + N_bar(:,1,idx_c)); 
            temp_c = temp_c + P_N_c(idx_c) * theta_c(idx_max);
        end
              
        temp_m = temp_m + binopdf(n_m,N,theta_m) * temp_c;
        
    end
    
    Risk_a = Risk_a - theta_m*temp_m;
end


