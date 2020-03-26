%%%%%%%%%%

clear;

M = 5;
N = 10;


N_bar = N_bar_set(M,N);
L_set = size(N_bar,2);

n = -5:N+5;


%%% Numerical PMF and mean
N_max_set = max(N_bar);

CMF_emp = NaN(size(n));
for idx = 1:numel(n)
    CMF_emp(idx) = numel(find(N_max_set <= n(idx))) / L_set;
end
PMF_emp = CMF_emp - [0, CMF_emp(1:end-1)];

mu_emp = mean(N_max_set);



%%% Guessed PMF and its mean
CMF = ones(1,length(n));
for m = 1:M
    idx_n = find(n <= (floor(N/m)-1));
    for idx = idx_n      
        CMF(idx) = CMF(idx) - (-1)^(m-1) * nchoosek(M,m) * ...
            nchoosek(N-m*(n(idx)+1)+M-1,M-1) / L_set;
    end
end
PMF = CMF - [0, CMF(1:end-1)];

mu = sum(n.*PMF);









%%% Guessed mean expression
mu_an = -1;
for m = 1:M
    for nn = 0:floor(N/m)
        mu_an = mu_an + (-1)^(m-1) * nchoosek(M,m) / L_set * nchoosek(N-m*nn+M-1,M-1);
    end
end




%%% Guessed risk expression
Risk_an = 1;
for m = 1:M
    for nn = 0:floor(N/m)
        Risk_an = Risk_an - 1/(N+M) * (-1)^(m-1) * nchoosek(M,m) * ...
            nchoosek(N-m*nn+M-1,M-1) / L_set;
    end
end






%%% Guessed limiting PMF and its mean
del_t = 1/50000;
t = -0.1:del_t:1.1;

C_lim = zeros(1,length(t));
for m = M:-1:1
    idx = find(t >= 1/m);
    C_lim(idx) = C_lim(idx) + (-1)^(M-m) * nchoosek(M,m) * (m*t(idx)-1).^(M-1);
end
P_lim = (C_lim - [0, C_lim(1:end-1)]) / del_t;   % differentiation

mu_lim_ap = sum(t.*P_lim)*del_t;        % note integration approx error...



%%% Guessed limiting mean expression
mu_lim_an = 1/M*sum(1./(1:M));




%%%%% Results
(mu_emp+1)/(N+M)
(mu+1)/(N+M)

(mu_an+1)/(N+M)
1-Risk_an

mu_lim_ap
mu_lim_an





%%% Plots
figure(1); clf; 
subplot(2,1,1); plot(n,CMF,'bo',n,CMF_emp,'r.'); 
grid on; xlabel('N_{max}'); ylabel('CMF'); title(['M = ',num2str(M),' , N = ',num2str(N)]);
subplot(2,1,2); plot(n,PMF,'bo',n,PMF_emp,'r.'); 
% hold on; plot(N/2*[1,1],[0,max(C_max)],'g',N/3*[1,1],[0,max(C_max)],'g',N/4*[1,1],[0,max(C_max)],'g');
grid on; xlabel('N_{max}'); ylabel('PMF'); title(['\mu/N = ',num2str(mu/N)]); %title(['\Delta = ',num2str(sum(abs(PMF-PMF_emp)))]);


figure(2); clf; 
subplot(2,1,1); plot(n/N,CMF,'bo',t,C_lim,'r'); grid on; title('Infinite N');
subplot(2,1,2); plot(n/N,PMF*N,'bo',t,P_lim,'r'); grid on; title(['\mu_{lim}/N = ',num2str(mu_lim_ap)]);
% subplot(2,1,2); plot(t,C_lim - CMF,'bo'); grid on;


% % if M == 2
% % %     figure(2); clf;
% % %     plot(N_bar(1,:),N_bar(2,:),'bo');
% % %     grid on; xlabel('N_1'); ylabel('N_2');
% %     
% % %     figure(3); clf;
% % %     scatter(N_bar(1,:),N_bar(2,:),[],N_max,'filled');
% % % %     scatter3(N_bar(1,:),N_bar(2,:),N_max,[],N_max,'filled');
% % %     grid on; xlabel('N_1'); ylabel('N_{max}');
% % elseif M == 3
% % %     figure(2); clf;
% % %     plot3(N_bar(1,:),N_bar(2,:),N_bar(3,:),'bo');
% % %     grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_3');
% %     
% % %     figure(3); clf;
% % %     scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),[],N_max,'filled');
% % % %     scatter3(N_bar(1,:),N_bar(2,:),N_max,[],N_max,'filled');
% % %     grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_{max}');
% %     
% %     
% % %     for n = N
% % %         idx = find(N_max == n);
% % %         figure(4); clf;
% % % %         plot(N_bar(1,idx),N_bar(2,idx),'bo');
% % %         plot3(N_bar(1,idx),N_bar(2,idx),N_bar(3,idx),'bo');
% % %         grid on; xlabel('N_1'); ylabel('N_2'); title(num2str(n));
% % %         view([-30,30]); axis([0,N,0,N]); pause(1);
% % %     end
% % elseif M == 4
% % %     figure(2); clf;
% % %     plot3(N_bar(1,:),N_bar(2,:),N_bar(3,:),'bo');
% % %     grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_3');
% % %     
% % %     figure(3); clf;
% % %     scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),[],N_max,'filled');
% % %     grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_3');
% %     
% % %     for n = ceil(N/2):ceil(N/1)
% % %         idx = find(N_max == n);
% % %         figure(4); clf;
% % %         plot3(N_bar(1,idx),N_bar(2,idx),N_bar(3,idx),'bo');
% % %         grid on; xlabel('N_1'); ylabel('N_2'); zlabel('N_3'); title(num2str(n));
% % %         axis([0,N,0,N,0,N]); view([0,90]); pause(2);
% % %     end
% % end





