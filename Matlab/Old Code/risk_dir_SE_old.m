function Risk_a = risk_opt_SE_old(Y,alpha,N)

alpha_0 = sum(alpha(:));

temp = 0;
for idx_x = 1:size(alpha,2)
    alpha_x = sum(alpha(:,idx_x));

    var_x = (cell2mat(Y).^2)'*alpha(:,idx_x)/alpha_x - (cell2mat(Y)'*alpha(:,idx_x)/alpha_x).^2;

    temp = temp + alpha_x/alpha_0*(N*alpha_x+alpha_0*alpha_x+alpha_0)/(alpha_0+N)/(alpha_x+1)*var_x;

end
Risk_a = temp;