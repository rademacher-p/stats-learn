%%% Training count PMF moment check

clear all;

M = 5;
N = 14;


%%% Generate set of Sufficient Statistics
N_bar = N_bar_set(M,N);



%%% Check Count Statistic PMF
L_theta = 2;
Theta = N_bar_set(M,L_theta)/L_theta;
P_Theta = 1/size(Theta,2) * ones(1,size(Theta,2));      % Uniform prior distribution

L_N_bar = size(N_bar,2);
P_N_bar = zeros(1,L_N_bar);
for idx_n = 1:L_N_bar
    
    P_N_bar(idx_n) = sum(P_Theta.*factorial(N)/prod(factorial(N_bar(:,idx_n))).*...
        prod(Theta.^(N_bar(:,idx_n)*ones(1,size(Theta,2))),1));
    
end

P_N_bar_a = 1/L_N_bar * ones(1,L_N_bar);
error_pmf = sqrt(mean(abs(P_N_bar - P_N_bar_a).^2))

% figure(1); clf;
% if M == 2     
%     plot(N_bar(1,:),P_N_bar_a,'bo',N_bar(1,:),P_N_bar,'r*')
%     grid on; title(num2str(error_pmf));
% elseif M == 3
%     plot3(N_bar(1,:),N_bar(2,:),P_N_bar_a,'bo',N_bar(1,:),N_bar(2,:),P_N_bar,'r*')
%     grid on; title(num2str(error_pmf));
% end




%%% 1st Moment
mu_N = zeros(M,1);
for idx = 1:M
    mu_N(idx) = sum(P_N_bar_a.*N_bar(idx,:));
end

mu_N_a = N/M*ones(M,1);
error_mu_N = mu_N - mu_N_a


%%% 2nd moments
R_N = zeros(M,M);
for idx_1 = 1:M
    for idx_2 = 1:M
        R_N(idx_1,idx_2) = sum(P_N_bar_a.*N_bar(idx_1,:).*N_bar(idx_2,:));
    end
end

% % m2_N_a = N/M*(2*N+M-1)/(M+1);
% % mx_N_a = N/M*(N-1)/(M+1);

R_N_a = N/M/(M+1)*[(2*N+M-1)*eye(M) + (N-1)*(ones(M)-eye(M))];
error_R_N = R_N - R_N_a



%%% Expected Squared Error, infinite data
P_y = N_bar/N;
var_y = ((1:M).^2)*P_y - ((1:M)*P_y).^2;

risk_emp = sum(P_N_bar_a.*var_y)

% risk_a = 1/6*(M+1)*(2*M+1) - 1/4*M/N*(M+1)*(2*N+M-1)
risk_emp_a = M*(M-1)*(N-1) / (12*N)


%%% Expected Square Error, Optimal Learner
P_y = (N/(N+M)*N_bar/N + M/(N+M)*M^-1);
var_y = ((1:M).^2)*P_y - ((1:M)*P_y).^2;
risk_opt = sum(P_N_bar_a.*var_y)

risk_prior = (M+1)*(M-1)/12;
risk_data = M*(M-1)*(N-1) / (12*N);
risk_cross = (M+1)*(2*M+1)/6 - M*(M-1)*(N-1)/(12*N) - (M+1)^2/4;
risk_opt_a = M/(N+M)*risk_prior + N/(N+M)*risk_data + N*M/(N+M)^2*risk_cross



%%% N_bar marginal validation
% % N = 11;
% % 
% % M = 5;
% % d = 2;
% % 
% % a = N_bar_set(d+1,N); a = a(1:d,:);
% % x = zeros(1,size(a,2));
% % for idx = 1:size(a,2)
% % %     x(idx) = nchoosek(N-a(idx)+M-1-d,M-1-d) / nchoosek(N+M-1,M-1);
% %     x(idx) = nchoosek(N-sum(a(:,idx))+M-1-d,M-1-d) / nchoosek(N+M-1,M-1);
% % end
% % 
% % % figure(10); clf; plot(a(1,:),x,'b*'); 
% % figure(10); clf; plot3(a(1,:),a(2,:),x,'b*'); 
% % grid on; title(num2str(sum(x)));
% % 
% % return






