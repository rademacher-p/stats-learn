%%%

clear;

M = 1:25;

% LB = M.*(M-1)/12;
% UB = (M+1).*(M-1)/12;

LB = (M-1)./(12*M);             % UNIT INTERVAL
UB = (M+1).*(M-1)./(12*M.^2);

% LB = M./(12*(M-1));       % UNIT INTERVAL mod
% UB = (M+1)./(12*(M-1));

figure(1); clf;
plot(M,UB,'.',M,LB,'.');
grid on; 
xlabel('$|\mathcal{Y}|$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
title('SE Risk, $\alpha = 1$','Interpreter','latex');
legend({'$N = 0$','$N/|\mathcal{X}| \to \infty$'},'Interpreter','latex','Location','southeast');



% return


% M = (1:5);
M = [2,5,10];

N = (0:50)';

% risk_opt_a = zeros(length(N),length(M));

% for idx_m = 1:length(M)
%     for idx_n = 1:length(N)
%         
%         risk_prior = (M+1)*(M-1)/12;
%         risk_data = M*(M-1)*(N-1) / (12*N);
%         risk_cross = (M+1)*(2*M+1)/6 - M*(M-1)*(N-1)/(12*N) - (M+1)^2/4;
% 
%         risk_opt_a = M/(N+M)*risk_prior + N/(N+M)*risk_data + N*M/(N+M)^2*risk_cross
%         risk_prior_emp = M/(N+M)*risk_prior + N/(N+M)*(risk_data + risk_cross)
%         risk_data_emp = M/(N+M)*(risk_prior + risk_cross) + N/(N+M)*risk_data
%         
%     end
% end


MM = ones(size(N))*M;
NN = N*ones(size(M));

% risk_opt_a = MM./(NN+MM).*risk_prior + NN./(NN+MM).*risk_data + NN.*MM./(NN+MM).^2.*risk_cross;
% risk_opt_a = MM./(NN+MM).*((MM+2).*(MM-1)/12) + NN./(NN+MM).*risk_data;

risk_opt = MM.*(MM-1).*(NN+MM+1)./(12*(NN+MM));

% risk_opt = (MM-1)./(12*MM).*(NN+MM+1)./(NN+MM); % UNIT INTERVAL

% risk_opt = zeros(numel(N),numel(M));
% for idx_N = 1:numel(N)
%     for idx_M = 1:numel(M)
%         risk_opt(idx_N,idx_M) = risk_opt_SE(num2cell((1:M(idx_M))'/M(idx_M)),ones(M(idx_M),1),N(idx_N));
%     end
% end


% risk_opt = MM./(12*(MM-1)).*(NN+MM+1)./(NN+MM); % UNIT INTERVAL mod


% risk_prior = (MM+1).*(MM-1)/12;
% risk_data = MM.*(MM-1).*(NN-1) ./ (12*NN);
% 
% % risk_cross = (MM+1).*(2*MM+1)/6 - MM.*(MM-1).*(NN-1)./(12*NN) - (MM+1).^2/4;
% risk_cross = (MM-1).*(1+MM./NN)/12;
% 
% risk_prior_emp = MM./(NN+MM).*risk_prior + NN./(NN+MM).*(risk_data + risk_cross);
% risk_data_emp = MM./(NN+MM).*(risk_prior + risk_cross) + NN./(NN+MM).*risk_data;



% for idx_m = 1:length(M)
%     figure(idx_m); clf;
%     plot(N,risk_opt_a(:,idx_m),N,risk_prior_emp(:,idx_m),N,risk_data_emp(:,idx_m));
%     grid on; xlabel('N'); ylabel('Risk'); title(['M = ',num2str(M(idx_m))]);
%     legend('Optimum','Prior','Data');
% end



figure(2); clf;
plot(N,risk_opt,'.-');
grid on; 
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
title('Squared-Error Risk');
str_leg = cell(1,length(M));
for idx = 1:length(M)
    str_leg(idx) = {['$M = ',num2str(M(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','southeast');







