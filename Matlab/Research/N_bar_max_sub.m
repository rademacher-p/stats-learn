%%%%%%%%%%

clear;

M = 6;
N = 3;


N_bar = N_bar_set(M,N);
L_set = size(N_bar,2);


N_max_set = zeros(M,L_set);
for m = 1:M
%     N_max_set(m,:) = max(N_bar(1:m,:),[],1);
    N_max_set(m,:) = max(N_bar(randperm(M,m),:),[],1);
end



n = -5:N+5;


%%% Numerical PMFs and means
CMF_emp = zeros(M,length(n));
for m = 1:M
    for idx = 1:length(n)
        CMF_emp(m,idx) = numel(find(N_max_set(m,:) <= n(idx))) / L_set;
    end
end
PMF_emp = CMF_emp - [zeros(M,1), CMF_emp(:,1:end-1)];

mu_emp = mean(N_max_set,2);




%%% Guessed PMFs and its means
CMF = zeros(M,length(n));
for k = 1:M
    for m = 0:k
        idx_n = find((n >= 0) & (n <= (floor(N/m)-1)));
        for idx = idx_n
            CMF(k,idx) = CMF(k,idx) - (-1)^(m-1) * nchoosek(k,m) * ...
                nchoosek(N-m*(n(idx)+1)+M-1,M-1) / L_set;
        end
    end
end
PMF = CMF - [zeros(M,1), CMF(:,1:end-1)];

mu = PMF*n';




%%% Guessed mean expression
mu_an = -1*ones(M,1);
for k = 1:M
    for m = 1:k
        idx_n = find((n >= 0) & (n <= floor(N/m)));
        for idx = idx_n
            mu_an(k) = mu_an(k) + (-1)^(m-1) * nchoosek(k,m) * nchoosek(N-m*n(idx)+M-1,M-1) / L_set;
%             mu_an(k) = mu_an(k) + (-1)^(m-1) * nchoosek(k,m) * prod(1 - m*n(idx)./(N+(1:M-1)));
        end
    end
end


    

%%% Guessed 1-Risk expression
P_c = zeros(M,1);
for k = 1:M
    for m = 1:k
        idx_n = find((n >= 0) & (n <= floor(N/m)));
        for idx = idx_n
            P_c(k) = P_c(k) + 1/(N+M) * (-1)^(m-1) * nchoosek(k,m) * nchoosek(N-m*n(idx)+M-1,M-1) / L_set;
%             P_c(k) = P_c(k) + 1/(N+M) * (-1)^(m-1) * nchoosek(k,m) * prod(1 - m*n(idx)./(N+(1:M-1)));
        end
    end
end





%%% Guessed limiting PMF and its mean
del_lim = 1/50000;

t = -0.1:del_lim:1.1;

CMF_lim = zeros(M,length(t));
for k = 1:M
    for m = 0:k
        idx = find((t >= 0) & (t < 1/m));
        CMF_lim(k,idx) = CMF_lim(k,idx) + (-1)^m * nchoosek(k,m) * (1-m*t(idx)).^(M-1);
    end
end
PMF_lim = (CMF_lim - [zeros(M,1), CMF_lim(:,1:end-1)]) / del_lim;   % differentiation

mu_lim_ap = del_lim*PMF_lim*t';           % integration approximation



%%% Guessed limiting mean expression
mu_lim_an = 1/M*cumsum(1./(1:M))';



%%%%% Results
(mu_emp+1)*M./(1:M)'/(N+M)
(mu+1)*M./(1:M)'/(N+M)

(mu_an+1)*M./(1:M)'/(N+M)
P_c*M./(1:M)'


mu_lim_ap*M./(1:M)'
mu_lim_an*M./(1:M)'



%%% Plots
figure(1); clf;
subplot(2,1,1);
plot(n,CMF_emp,'bo',n,CMF,'r*');
grid on; set(gca,'YLim',[0,1]);
subplot(2,1,2);
plot(n/N,CMF_emp,'bo',t,CMF_lim,'r');
grid on; set(gca,'YLim',[0,1]);

figure(2); clf;
subplot(2,1,1);
plot(n,CMF-CMF_emp,'o');
grid on;
subplot(2,1,2);
plot(n,PMF-PMF_emp,'o');
grid on;



% figure(1); clf; 
% subplot(2,1,1); plot(0:N,CMF(M,:),'bo',0:N,CMF_emp(end,:),'r.'); 
% grid on; xlabel('N_{max}'); ylabel('CMF'); title(['M = ',num2str(M),' , N = ',num2str(N)]);
% subplot(2,1,2); plot(0:N,PMF(M,:),'bo',0:N,PMF_emp(end,:),'r.'); 
% % hold on; plot(N/2*[1,1],[0,max(C_max)],'g',N/3*[1,1],[0,max(C_max)],'g',N/4*[1,1],[0,max(C_max)],'g');
% grid on; xlabel('N_{max}'); ylabel('PMF'); %title(['\mu = ',num2str(mu)]); %title(['\Delta = ',num2str(sum(abs(PMF-PMF_emp)))]);
% 
% 
% figure(2); clf; 
% subplot(2,1,1); plot((0:N)/N,CMF(M,:),'bo',t,CMF_lim(M,:),'r'); grid on; title('Infinite N');
% subplot(2,1,2); plot((0:N)/N,PMF(M,:)*N,'bo',t,PMF_lim(M,:),'r'); grid on; title(['\mu_{lim}/N = ',num2str(mu_lim_ap)]);
% % subplot(2,1,2); plot(t,C_lim - CMF,'bo'); grid on;










