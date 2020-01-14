%%%%%%%%%%

clear;
clc;


temp = 1/3*betainc(1/2,2,2)-1/3*(betainc(2/3,2,2)-betainc(1/2,2,2))...
    +2*2/3*(betainc(2/3,1,3)-betainc(1/2,1,3));
3*temp

1/3*sum(1./(1:3))



% N = 1000;

% alpha = ones(2,1);
alpha = [7.3;1;1];


%
M = length(alpha);
alpha_0 = sum(alpha);

N_s = 100000;

temp = zeros(M,N_s);
for m = 1:M
    temp(m,:) = gamrnd(alpha(m),1,[1,N_s]);
end
theta = temp ./ (ones(M,1)*sum(temp,1));

theta_max = max(theta,[],1);



mu_emp = mean(theta_max)
if M == 2
%     mu_an = alpha(1)/alpha_0*betainc(1/2,alpha(2),alpha(1)+1) + ...
%         alpha(2)/alpha_0*betainc(1/2,alpha(1),alpha(2)+1);
    
    mu_an = alpha(1)/alpha_0*(betainc(1,alpha(1)+1,alpha(2)) - betainc(1/2,alpha(1)+1,alpha(2))) + ...
        alpha(2)/alpha_0*(betainc(1,alpha(2)+1,alpha(1)) - betainc(1/2,alpha(2)+1,alpha(1)))
    
    mu_sum = 0;
    for n = 0:300
        mu_sum = mu_sum + (1/2)^n *(beta(alpha(1),alpha(2)+1+n)^(-1) + beta(alpha(2),alpha(1)+1+n)^(-1));
    end
    mu_sum = mu_sum*(1/2)^(alpha_0+1)/alpha_0
    
elseif M == 3
    
    A_p = perms(alpha).';
    
    for idx_p = 1:size(A_p,2)
        a_p = A_p(:,idx_p);
        
        temp_1 = a_p(3)/alpha_0*(betainc(1,a_p(3)+1,sum(a_p(1:2))) ...
            - 2*betainc(1/2,a_p(3)+1,sum(a_p(1:2))) ...
            + betainc(1/3,a_p(3)+1,sum(a_p(1:2))));
        
        
        
        
        
    end
    
%     temp1 = alpha(3)/alpha_0*betainc(1/3,alpha(1),alpha(2)+alpha(3)+1)*betainc(1/2,alpha(2),alpha(3)+1);
%     temp2 = alpha(3)/alpha_0*betainc(1/3,alpha(2),alpha(1)+alpha(3)+1)*betainc(1/2,alpha(1),alpha(3)+1);
%     
%     temp3 = 0;
%     N_sum = 100;
%     for n = 0:N_sum
%         for m = 0:N_sum
% %             temp3 = temp3 + (1/3)^(n+m) / exp(gammaln(alpha(1)+1+n)+...
% %                 gammaln(alpha(2)+1+m)+gammaln(alpha(3))-gammaln(alpha_0+2+n+m));
%             
%             temp3 = temp3 + (1/3)^(n+m) / (alpha_0+1+n+m) / exp(gammaln(alpha(1)+1+n)+...
%                 gammaln(alpha(2)+1+m)+gammaln(alpha(3))-gammaln(alpha_0+2+n+m));
%         end
%     end
% %     temp3 = temp3*(1/3)^(alpha(1)+alpha(2))*(2/3)^(alpha(3)+1)/alpha_0/(alpha_0+1);
%     temp3 = temp3*(1/3)^(alpha(1)+alpha(2))*(2/3)^(alpha(3)+1)/alpha_0;
% 
%     mu_an = temp1 + temp2 - temp3;
%     
%     mu_an = 3*mu_an
    
    
end



% del_t = 0.001;
% t = -0.1:del_t:1.1;
% 
% CDF_emp = zeros(size(t));
% for idx_t = 1:length(t)
%     CDF_emp(idx_t) = sum(theta_max <= t(idx_t))/N_s;
% end
% 
% % CDF_a = zeros(size(t));
% % for idx_t = 1:length(t)
% %     CDF_a(idx_t) = 1 - betainc(1-t,alpha(1),alpha(2))
% % end
% 
% 
% figure(10); clf;
% plot(t,CDF_emp);
% grid on;


return



N_bar = N_bar_set(M,N);
L_set = size(N_bar,2);

n = -5:N+5;


%%%
P_N_bar = zeros(1,L_set);
for idx = 1:L_set
    P_N_bar(idx) = factorial(N)*gamma(alpha_0)/gamma(N+alpha_0)*...
        prod(gamma(N_bar(:,idx)+alpha))/prod(gamma(alpha))/prod(factorial(N_bar(:,idx)));
end


%%% Numerical PMF and mean
N_max_set = max(N_bar);

CMF_emp = zeros(1,length(n));
for idx = 1:length(n)
%     CMF_emp(idx) = numel(find(N_max_set <= n(idx))) / L_set;
    CMF_emp(idx) = sum(P_N_bar(find(N_max_set <= n(idx))));
end
PMF_emp = CMF_emp - [0, CMF_emp(1:end-1)];

mu_emp = mean(N_max_set);



%%% Plots
figure(1); clf; 
subplot(2,1,1); plot(n,CMF_emp,'bo'); 
grid on; xlabel('N_{max}'); ylabel('CMF'); title(['M = ',num2str(M),' , N = ',num2str(N)]);
% idx_plot = find(PMF_emp>0);
idx_plot = 1:numel(n);
subplot(2,1,2); plot(n(idx_plot),PMF_emp(idx_plot),'bo'); 
grid on; xlabel('N_{max}'); ylabel('PMF'); title(['\mu = ',num2str(mu_emp)]); %title(['\Delta = ',num2str(sum(abs(PMF-PMF_emp)))]);


