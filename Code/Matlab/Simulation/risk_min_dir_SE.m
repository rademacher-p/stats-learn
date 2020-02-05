function Risk_a = risk_min_dir_SE(Y,X,N,alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Min. Bayes Risk, Dirichlet Prior, SE Loss
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha_0 = sum(alpha(:));

temp = 0;
for idx_x = 1:size(alpha,2)
    alpha_x = sum(alpha(:,idx_x));
    
% %     if alpha_x == 0
% %         continue 
% %     end

    var_x = (cell2mat(Y).^2)'*alpha(:,idx_x)/alpha_x - (cell2mat(Y)'*alpha(:,idx_x)/alpha_x).^2;

    temp = temp + alpha_x/alpha_0*(N*alpha_x+alpha_0*alpha_x+alpha_0)/(alpha_0+N)/(alpha_x+1)*var_x;

end
Risk_a = temp;