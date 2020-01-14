%%%

clear;

M = 1:100;

P_c = cumsum(1./M) ./ M;    
    
% figure(3); clf;
% plot(M,P_c,'b.',M,log(M)./M,'r');
% grid on; xlabel('M'); ylabel('R(f^*)');

LB = 1 - P_c;
UB = 1 - 1./M;

figure(1); clf;
plot(M,UB,'.',M,LB,'.');
grid on; 
xlabel('$|\mathcal{Y}|$','Interpreter','latex'); 
% xlabel('$M$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*$','Interpreter','latex'); 
title('0-1 Risk Bounds','Interpreter','latex');
legend({'$N = 0$','$N \to \infty$'},'Interpreter','latex','Location','southeast');


% figure(4); clf;
% plot(M,LB./UB - 1,'b.');
% grid on;


return


%%% 

MM = [2,4,8];
NN = 0:25;

Risk = zeros(length(NN),length(MM));

for idx_M = 1:length(MM)
    
    M = MM(idx_M);
    
    for idx_N = 1:length(NN)
        
        N = NN(idx_N);
        
%         Risk(idx_N,idx_M) = -1/(N+M);
%         for m = 1:M
%             for n = ceil((N+M)/m):(N+M)
%                 Risk(idx_N,idx_M) = Risk(idx_N,idx_M) + ...
%                     -1/(N+M) * (-1)^m * nchoosek(M,m) * prod(1 - m*n./(N+(1:M-1)));
%             end
%         end  
        
        Risk(idx_N,idx_M) = risk_opt_01((1:M)',ones(M,1),N);
        
    end
end

figure(2); clf;
plot(NN,Risk,'.-');
grid on; 
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
title('0-1 Risk');
str_leg = cell(1,length(MM));
for idx = 1:length(MM)
    str_leg(idx) = {['$M = ',num2str(MM(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','southeast');







