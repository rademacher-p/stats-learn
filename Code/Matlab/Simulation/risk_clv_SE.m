function Risk = risk_clv_SE(Y,X,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clairvoyant Risk, Squared-Error Loss
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y_mat = cell2mat(Y);

Risk = 0;
for idx_x = 1:numel(X)
    
    theta_m = sum(theta(:,idx_x));
    if theta_m == 0
        continue
    end
    
    theta_c = theta(:,idx_x) ./ theta_m;

    mu_theta_c = Y_mat'*theta_c;
    var_theta_c = (Y_mat.^2)'*theta_c - mu_theta_c^2;
        
    Risk = Risk + theta_m*var_theta_c;
end