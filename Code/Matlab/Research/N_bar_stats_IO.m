%%%%%

clear;

M = 6;
L = 2;

N = 14;


%%%
temp = N_bar_set(L+1,N);
N_sub = temp(1:end-1,:);

%
L_set = nchoosek(N+M-1,N);
L_set_sub = size(N_sub,2);

P_N_sub = zeros(1,L_set_sub);
for idx = 1:L_set_sub
    P_N_sub(idx) = nchoosek(N-sum(N_sub(:,idx))+M-1-L,M-1-L) / L_set;
end


figure(10); clf;
scatter(N_sub(1,:),N_sub(2,:),[],P_N_sub,'filled');
grid on;



temp1 = repmat(reshape(N_sub+1,[L,1,L_set_sub]),[1,L]);
temp2 = repmat(reshape((N+M)*sum(N_sub+1),[1,1,L_set_sub]),[L,L]);
dat = temp1.*permute(temp1,[2,1,3])./temp2;

moments = sum(dat .* repmat(reshape(P_N_sub,[1,1,L_set_sub]),[L,L]),3);

% ex1 = 0;
% ex2 = 0;
% for k = 0:N
%     ex1 = ex1 + 1/(N+M)/L_set/L/(L+1)*nchoosek(N-k+M-1-L,M-1-L)*nchoosek(k+L-1,k)*(k+L+1);
%     ex2 = ex2 + 1/(N+M)/L_set/L/(L+1)*nchoosek(N-k+M-1-L,M-1-L)*nchoosek(k+L-1,k)*(2*k+L+1);
% end
% ex = eye(L)*ex1  + (ones(L)-eye(L))*ex2;


exx1 = 1/(N+M)/M/L/(L+1)*((N+M)*L+M);
exx2 = 1/(N+M)/M/L/(L+1)*((2*N+M)*L+M);
exx = eye(L)*exx2  + (ones(L)-eye(L))*exx1;



moments
exx

return

%%
a = 7;
b = 3;
N = 15;

x = 0;
for k = 0:N
    x = x + nchoosek(k+a,a)*nchoosek(N-k+b,b);
end

z = nchoosek(N+a+b+1,N);


% x
% z


x = 0;
for k = 0:N+1
    x = x + nchoosek(k+a,a)*nchoosek(N+1-k+b,b);
end


z = nchoosek(N+1+a,a);
for l = 0:b
    z = z + nchoosek(N+a+l+1,N);
end

% z = nchoosek(N+a+b+2,N+1);

x
z


%%

% figure(11); clf;
% % scatter(N_sub(1,:),N_sub(2,:),[],squeeze(dat(1,1,:)),'filled');
% plot3(N_sub(1,:),N_sub(2,:),squeeze(dat(1,1,:)),'bo');
% grid on;
% 
% figure(12); clf;
% % scatter(N_sub(1,:),N_sub(2,:),[],squeeze(dat(1,2,:)),'filled');
% plot3(N_sub(1,:),N_sub(2,:),squeeze(dat(1,2,:)),'bo');
% grid on;

