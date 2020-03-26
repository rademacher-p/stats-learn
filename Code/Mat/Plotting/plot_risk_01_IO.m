%%%

clear;

% M = 1:100;
% 
% P_c = cumsum(1./M) ./ M;    
%     
% % figure(1); clf;
% % plot(M,P_c,'b.',M,log(M)./M,'r');
% % grid on; xlabel('M'); ylabel('R(f^*)');
% 
% LB = 1 - P_c;
% UB = 1 - 1./M;
% 
% figure(2); clf;
% plot(M,UB,'b.',M,LB,'r.');
% grid on; 
% xlabel('$M$','Interpreter','latex'); 
% ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
% title('0-1 Risk');
% legend({'$N = 0$','$N \to \infty$'},'Interpreter','latex','Location','southeast');
% 
% 
% % figure(4); clf;
% % plot(M,LB./UB - 1,'b.');
% % grid on;
% 
% 
% % return


%%% 

MM_y = [2,4,8];
MM_x = [1,2,4];
NN = 0:100;
% NN = [0,5,10,20];

Risk = zeros(length(NN),length(MM_y),length(MM_x));

for idx_M_y = 1:length(MM_y)
    
    M_y = MM_y(idx_M_y);

    for idx_M_x = 1:length(MM_x)
    
        M_x = MM_x(idx_M_x);
        M = M_y*M_x;
        
        for idx_N = 1:length(NN)
        
            N = NN(idx_N);
            
%             %%%
%             Risk(idx_N,idx_M_y,idx_M_x) = 1;
%             for m = 1:M_y
%                 for n = 0:ceil((N+M)/m)-1
%                     Risk(idx_N,idx_M_y,idx_M_x) = Risk(idx_N,idx_M_y,idx_M_x) + ...
%                         M_x/(N+M) * (-1)^m * nchoosek(M_y,m) * prod(1 - m*n./(N+(1:M-1)));
%                 end
%             end

            Risk(idx_N,idx_M_y,idx_M_x) = risk_opt_01((1:M_y)',ones(M_y,M_x),N);

        end
    end
end



Y_lims = [0,1];
% Y_lims = [0.2,0.5];

mark_size = 10;


idx_M_x = find(MM_x == 1);
figure(1); clf;
plot(NN,Risk(:,:,idx_M_x),'.','MarkerSize',mark_size);
grid on; set(gca,'YLim',Y_lims);
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*$','Interpreter','latex'); 
title(['0-1 Risk, $|\mathcal{X}|=',num2str(MM_x(idx_M_x)),'$'],'Interpreter','latex');
% title(['0-1 Risk, $M_x=',num2str(MM_x(idx_M_x)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_y));
for idx = 1:length(MM_y)
    str_leg(idx) = {['$|\mathcal{Y}| = ',num2str(MM_y(idx)),'$']};
%     str_leg(idx) = {['$M_y = ',num2str(MM_y(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','northeast');


idx_M_y = find(MM_y == 2);
figure(2); clf;
plot(NN,squeeze(Risk(:,idx_M_y,:)),'.','MarkerSize',mark_size);
grid on; set(gca,'YLim',Y_lims);
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*$','Interpreter','latex'); 
title(['0-1 Risk, $|\mathcal{Y}|=',num2str(MM_y(idx_M_y)),'$'],'Interpreter','latex');
% title(['0-1 Risk, $M_y=',num2str(MM_y(idx_M_y)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_x));
for idx = 1:length(MM_x)
    str_leg(idx) = {['$|\mathcal{X}| = ',num2str(MM_x(idx)),'$']};
%     str_leg(idx) = {['$M_x = ',num2str(MM_x(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','northeast');

idx_M_y = find(MM_y == 2);
figure(10); clf;
plot(NN'./MM_x,squeeze(Risk(:,idx_M_y,:)),'.','MarkerSize',mark_size);
grid on; set(gca,'YLim',Y_lims);
set(gca,'XLim',[0,10]);
xlabel('$N/|\mathcal{X}|$','Interpreter','latex'); 
% xlabel('$N/M_x$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*$','Interpreter','latex'); 
title(['0-1 Risk, $|\mathcal{Y}|=',num2str(MM_y(idx_M_y)),'$'],'Interpreter','latex');
% title(['0-1 Risk, $M_y=',num2str(MM_y(idx_M_y)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_x));
for idx = 1:length(MM_x)
    str_leg(idx) = {['$|\mathcal{X}| = ',num2str(MM_x(idx)),'$']};
%     str_leg(idx) = {['$M_x = ',num2str(MM_x(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','northeast');


idx_N = find(NN == 10);
figure(3); clf;
plot(MM_y,squeeze(Risk(idx_N,:,:)),'.');
grid on; set(gca,'YLim',Y_lims);
xlabel('$M_y$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*$','Interpreter','latex'); 
title(['0-1 Risk, $N=',num2str(NN(idx_N)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_x));
for idx = 1:length(MM_x)
    str_leg(idx) = {['$M_x = ',num2str(MM_x(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','southeast');


idx_N = find(NN == 10);
figure(4); clf;
plot(MM_x,squeeze(Risk(idx_N,:,:))','.');
grid on; set(gca,'YLim',Y_lims);
xlabel('$M_x$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*$','Interpreter','latex'); 
title(['0-1 Risk, $N=',num2str(NN(idx_N)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_y));
for idx = 1:length(MM_y)
    str_leg(idx) = {['$M_y = ',num2str(MM_y(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','southeast');



idx_N = find(NN == 10);
idx_M_y = find(MM_y == 2);
figure(5); clf;
plot(MM_x,squeeze(Risk(:,idx_M_y,:))','.');
grid on; set(gca,'YLim',Y_lims);
xlabel('$M_x$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*$','Interpreter','latex'); 
title(['0-1 Risk, $M_y=',num2str(MM_y(idx_M_y)),'$'],'Interpreter','latex');
str_leg = cell(1,length(NN));
for idx = 1:length(NN)
    str_leg(idx) = {['$N = ',num2str(NN(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','southeast');




% figure(4); clf;
% plot(NN,Risk,'.-');
% grid on; 
% xlabel('$N$','Interpreter','latex'); 
% ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
% title('0-1 Risk');
% str_leg = cell(1,length(MM));
% for idx = 1:length(MM)
%     str_leg(idx) = {['$M = ',num2str(MM(idx)),'$']};
% end
% legend(str_leg,'Interpreter','latex','Location','southeast');







