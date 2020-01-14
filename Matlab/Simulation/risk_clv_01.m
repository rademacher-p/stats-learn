function Risk = risk_clv_01(Y,X,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clairvoyant Risk, 0-1 Loss
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Risk = 1;
for idx_x = 1:numel(X)
    Risk = Risk - max(theta(:,idx_x));
end