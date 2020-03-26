%%%

clear;

NMx = 0:100;
My = [2,5,10];

R = zeros(numel(NMx),numel(My));

for i_1 = 1:numel(NMx)
    for i_2 = 1:numel(My)
%         R(i_1,i_2) = My(i_2)*(My(i_2)-1)*(NMx(i_1)+My(i_2)+1)/12/(NMx(i_1)+My(i_2));
        R(i_1,i_2) = (My(i_2)-1)*(NMx(i_1)+My(i_2)+1)/12/My(i_2)/(NMx(i_1)+My(i_2));
    end    
end


figure(20); clf;
plot(NMx,R,'-');
grid on; 
xlabel('$N/M_x$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
title('Squared-Error Risk','Interpreter','latex');
legend({'$N = 0$','$N \to \infty$'},'Interpreter','latex','Location','southeast');
str_leg = cell(1,length(My));
for i = 1:length(My)
    str_leg(i) = {['$M_y = ',num2str(My(i)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','northeast');




%%% 

MM_y = [2:8];
MM_x = [1,2,4,8];
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
                       
            Risk(idx_N,idx_M_y,idx_M_x) = risk_opt_SE(num2cell((1:M_y)'/M_y),ones(M_y,M_x),N);

        end
    end
end




idx_M_x = find(MM_x == MM_x(1));
figure(1); clf;
plot(NN,Risk(:,:,idx_M_x),'.');
grid on; set(gca,'YLim',[0,0.1]);
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f^*)$','Interpreter','latex'); 
title(['Squared-Error Risk, $M_x=',num2str(MM_x(idx_M_x)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_y));
for idx = 1:length(MM_y)
    str_leg(idx) = {['$M_y = ',num2str(MM_y(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','northeast');


idx_M_y = find(MM_y == 2);
figure(2); clf;
plot(NN,squeeze(Risk(:,idx_M_y,:)),'.');
grid on; set(gca,'YLim',[0,0.1]);
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f^*)$','Interpreter','latex'); 
title(['Squared-Error Risk, $M_y=',num2str(MM_y(idx_M_y)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_x));
for idx = 1:length(MM_x)
    str_leg(idx) = {['$M_x = ',num2str(MM_x(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','northeast');

idx_M_y = find(MM_y == 2);
figure(10); clf;
plot(NN'./MM_x,squeeze(Risk(:,idx_M_y,:)),'.');
grid on; set(gca,'YLim',[0,0.1]);
xlabel('$N/M_x$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f^*)$','Interpreter','latex'); 
title(['Squared-Error Risk, $M_y=',num2str(MM_y(idx_M_y)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_x));
for idx = 1:length(MM_x)
    str_leg(idx) = {['$M_x = ',num2str(MM_x(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','northeast');


idx_N = find(NN == 10);
figure(3); clf;
plot(MM_y,squeeze(Risk(idx_N,:,:)),'.');
grid on; set(gca,'YLim',[0,0.1]);
xlabel('$M_y$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f^*)$','Interpreter','latex'); 
title(['Squared-Error Risk, $N=',num2str(NN(idx_N)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_x));
for idx = 1:length(MM_x)
    str_leg(idx) = {['$M_x = ',num2str(MM_x(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','southeast');


idx_N = find(NN == 10);
figure(4); clf;
plot(MM_x,squeeze(Risk(idx_N,:,:))','.');
grid on; set(gca,'YLim',[0,0.1]);
xlabel('$M_x$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f^*)$','Interpreter','latex'); 
title(['Squared-Error Risk, $N=',num2str(NN(idx_N)),'$'],'Interpreter','latex');
str_leg = cell(1,length(MM_y));
for idx = 1:length(MM_y)
    str_leg(idx) = {['$M_y = ',num2str(MM_y(idx)),'$']};
end
legend(str_leg,'Interpreter','latex','Location','southeast');



idx_N = find(NN == 10);
idx_M_y = find(MM_y == 2);
figure(5); clf;
plot(MM_x,squeeze(Risk(:,idx_M_y,:))','.');
grid on; set(gca,'YLim',[0,0.1]);
xlabel('$M_x$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f^*)$','Interpreter','latex'); 
title(['Squared-Error Risk, $M_y=',num2str(MM_y(idx_M_y)),'$'],'Interpreter','latex');
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







