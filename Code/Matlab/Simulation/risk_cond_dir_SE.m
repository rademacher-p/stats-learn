function Risk_a = risk_cond_dir_SE(Y,X,N,theta,alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conditional Risk, SE Loss, Dirichlet Learner
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y_mat = cell2mat(Y);

Risk_a = 0;

for idx_x = 1:numel(X)
    
    theta_m = sum(theta(:,idx_x));
    theta_c = theta(:,idx_x) ./ theta_m;

    alpha_m = sum(alpha(:,idx_x));
    alpha_c = alpha(:,idx_x) ./ alpha_m;
    
    
    mu_theta_c = Y_mat'*theta_c;
    var_theta_c = (Y_mat.^2)'*theta_c - mu_theta_c^2;
    
    mu_alpha_c = Y_mat'*alpha_c;
    
    bias_alpha_sq = (mu_alpha_c - mu_theta_c)^2;
    
    temp = 0;
    for n = 0:N
        temp = temp + binopdf(n,N,theta_m) ...
            * (var_theta_c*(1 + n/(alpha_m + n)^2) ...
            + bias_alpha_sq*(alpha_m/(alpha_m + n))^2);
    end   
    
    Risk_a = Risk_a + theta_m*temp;
end


