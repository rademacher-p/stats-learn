%%%%%%%%%%

clear;
clc;

N = 100;
% alpha = ones(2,1);
alpha = [3;1];


M = length(alpha);

N_bar = N_bar_set(M,N);
L_set = size(N_bar,2);


del_t = 0.00001;
t = -0.1:del_t:1.1;


%%%
alpha_0 = sum(alpha);

P_N_bar = zeros(1,L_set);
for idx = 1:L_set
    P_N_bar(idx) = factorial(N)*gamma(alpha_0)/gamma(N+alpha_0)*...
        prod(gamma(N_bar(:,idx)+alpha))/prod(gamma(alpha))/prod(factorial(N_bar(:,idx)));
end


%%%
P_y_N = (N_bar + alpha*ones(1,L_set))/(N + alpha_0); %%%%%%%%%% ??????????????
P_max_set = max(P_y_N);

CMF_emp = zeros(size(t));
for idx_t = 1:length(t)
    CMF_emp(idx_t) = sum(P_N_bar(find(P_max_set <= t(idx_t))));
end
PMF_emp = CMF_emp - [0, CMF_emp(1:end-1)];

mu_emp = mean(P_max_set);


figure(1); clf; 
subplot(2,1,1); plot(t,CMF_emp,'bo'); 
grid on; xlabel('N_{max}'); ylabel('CMF'); title(['M = ',num2str(M),' , N = ',num2str(N)]);
idx_plot = find(PMF_emp>0);
subplot(2,1,2); plot(t(idx_plot),PMF_emp(idx_plot),'bo'); 
grid on; xlabel('N_{max}'); ylabel('PMF'); title(['\mu = ',num2str(mu_emp)]); %title(['\Delta = ',num2str(sum(abs(PMF-PMF_emp)))]);



figure(3); clf;
scatter(N_bar(1,:),N_bar(2,:),[],P_N_bar,'filled');
% scatter3(N_bar(1,:),N_bar(2,:),N_max,[],N_max,'filled');
grid on; xlabel('N_1'); ylabel('N_{max}');

% figure(3); clf;
% scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),[],P_N_bar,'filled');
% grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_{max}');

return








%%%
n = 0:N;

CMF = zeros(1,N+1);
for m = M:-1:1
    for n = 0:N      
        if n >= (N+M-m)/m
            CMF(n+1) = CMF(n+1) + (-1)^(M-m) * nchoosek(M,m) * nchoosek(m*n-N+m-1,M-1) / L_set;
        end
    end
end
PMF = CMF - [0, CMF(1:end-1)];

mu = sum((0:N).*PMF);



%%%%%
N_lim = 50000;

t = (0:N_lim)/N_lim;

C_lim = zeros(1,N_lim+1);
for m = M:-1:1
    idx = find(t >= 1/m);
    C_lim(idx) = C_lim(idx) + (-1)^(M-m) * nchoosek(M,m) * (m*t(idx)-1).^(M-1);
end
P_lim = (C_lim - [0, C_lim(1:end-1)]) * N_lim;   % differentiation

mu_lim_ap = sum(t.*P_lim)/N_lim;        % note integration approx error...



%%%
mu_lim_an = 1/M*sum(1./(1:M));


mu_emp/N
mu/N

mu_lim_ap
mu_lim_an





%%% Plots
figure(1); clf; 
subplot(2,1,1); plot(0:N,CMF,'bo',0:N,CMF_emp,'r.'); 
grid on; xlabel('N_{max}'); ylabel('CMF'); title(['M = ',num2str(M),' , N = ',num2str(N)]);
subplot(2,1,2); plot(0:N,PMF,'bo',0:N,PMF_emp,'r.'); 
% hold on; plot(N/2*[1,1],[0,max(C_max)],'g',N/3*[1,1],[0,max(C_max)],'g',N/4*[1,1],[0,max(C_max)],'g');
grid on; xlabel('N_{max}'); ylabel('PMF'); title(['\mu/N = ',num2str(mu/N)]); %title(['\Delta = ',num2str(sum(abs(PMF-PMF_emp)))]);


figure(2); clf; 
subplot(2,1,1); plot((0:N)/N,CMF,'bo',t,C_lim,'r'); grid on; title('Infinite N');
subplot(2,1,2); plot((0:N)/N,PMF*N,'bo',t,P_lim,'r'); grid on; title(['\mu_{lim}/N = ',num2str(mu_lim_ap)]);
% subplot(2,1,2); plot(t,C_lim - CMF,'bo'); grid on;


% % if M == 2
% %     figure(2); clf;
% %     plot(N_bar(1,:),N_bar(2,:),'bo');
% %     grid on; xlabel('N_1'); ylabel('N_2');
% %     
% %     figure(3); clf;
% %     scatter(N_bar(1,:),N_bar(2,:),[],N_max,'filled');
% % %     scatter3(N_bar(1,:),N_bar(2,:),N_max,[],N_max,'filled');
% %     grid on; xlabel('N_1'); ylabel('N_{max}');
% % elseif M == 3
% %     figure(2); clf;
% %     plot3(N_bar(1,:),N_bar(2,:),N_bar(3,:),'bo');
% %     grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_3');
% %     
% %     figure(3); clf;
% % %     scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),[],N_max,'filled');
% %     scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),[],P_N_bar,'filled');
% % %     scatter3(N_bar(1,:),N_bar(2,:),N_max,[],N_max,'filled');
% %     grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_{max}');
% %     
% %     
% %     for n = N
% %         idx = find(N_max == n);
% %         figure(4); clf;
% % %         plot(N_bar(1,idx),N_bar(2,idx),'bo');
% %         plot3(N_bar(1,idx),N_bar(2,idx),N_bar(3,idx),'bo');
% %         grid on; xlabel('N_1'); ylabel('N_2'); title(num2str(n));
% %         view([-30,30]); axis([0,N,0,N]); pause(1);
% %     end
% % elseif M == 4
% %     figure(2); clf;
% %     plot3(N_bar(1,:),N_bar(2,:),N_bar(3,:),'bo');
% %     grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_3');
% %     
% %     figure(3); clf;
% %     scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),[],N_max,'filled');
% %     grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_3');
% %     
% %     for n = ceil(N/2):ceil(N/1)
% %         idx = find(N_max == n);
% %         figure(4); clf;
% %         plot3(N_bar(1,idx),N_bar(2,idx),N_bar(3,idx),'bo');
% %         grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_3'); title(num2str(n));
% %         axis([0,N,0,N,0,N]); view([0,90]); pause(2);
% %     end
% % end





