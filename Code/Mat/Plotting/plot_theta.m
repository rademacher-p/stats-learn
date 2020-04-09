%%%%%

clear;

alpha_0_vec = 3*[100,.01];

P_y = [1/3; 1/3; 1/3];
% P_y = [.2; .3; .5];


N_t = 121;

clims = [0,1];

m_size = [80];

Theta = N_bar_set(numel(P_y),N_t) / N_t;
L_set = size(Theta,2);


% %%%
% [~,idx_sort] = sort(max(Theta));
% Theta = Theta(:,idx_sort);
% %%%


figure(10); clf;
for idx_vec = 1:numel(alpha_0_vec)
    
alpha_0 = alpha_0_vec(idx_vec);
    
alpha = alpha_0*P_y;
PDF_dir = beta_multi(alpha)^-1 * prod(Theta.^((alpha-1)*ones(1,L_set)));


% PDF_dir = -1*sum(Theta .* log2(Theta)); %%% ENTROPY

% PDF_dir = NaN(1,L_set);   %%% Lp NORMS
% for kk = 1:L_set
%     PDF_dir(:,kk) = norm(Theta(:,kk),.5);
% %     PDF_dir(:,kk) = max(Theta(:,kk));
% %     PDF_dir(:,kk) = sum(Theta(:,kk)>0);
% end




sum(PDF_dir) / (N_t^(numel(P_y)-1))
pause

ii = find(isinf(PDF_dir)) %%%%%%%%%%
% Theta(:,ii);
Theta_plot = Theta;

Theta_plot(:,ii) = [];
PDF_dir(ii) = [];

% m_size = 50 + 50*PDF_dir / max(PDF_dir);

subplot(1,numel(alpha_0_vec),idx_vec)
% scatter3(Theta_plot(1,:),Theta_plot(2,:),Theta_plot(3,:),m_size(idx_vec),PDF_dir,'.');
scatter3(Theta_plot(1,:),Theta_plot(2,:),Theta_plot(3,:),m_size,PDF_dir,'.');

grid on; 
axis equal; 
view(135,45); 

xlabel('$\theta(\mathcal{Y}_1,\mathcal{X}_1)$','Interpreter','latex'); 
ylabel('$\theta(\mathcal{Y}_2,\mathcal{X}_1)$','Interpreter','latex'); 
zlabel('$\theta(\mathcal{Y}_3,\mathcal{X}_1)$','Interpreter','latex'); 
% xlabel('$\tilde{\theta}(\mathcal{Y}_1;x)$','Interpreter','latex'); 
% ylabel('$\tilde{\theta}(\mathcal{Y}_2;x)$','Interpreter','latex');
% zlabel('$\tilde{\theta}(\mathcal{Y}_3;x)$','Interpreter','latex');

% vec_str_P = num2str(P_y','%0.2f,');
% title(['$\mathrm{P}_{\mathrm{y},\mathrm{x}} = [',vec_str_P(1:end-1),...
%     ']^\mathrm{T}$, $\alpha_0 = ',num2str(alpha_0),'$'],'Interpreter','latex');

str_a = num2str(alpha','%g,');
title(['$\alpha = [',str_a(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
% title(['$\alpha(\cdot,x) = [',str_a(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');



cbar = colorbar; cbar.Label.Interpreter = 'latex'; 
cbar.Label.String = '$\mathrm{p}(\theta)$'; 
% cbar.Label.String = '$\mathrm{p}_{\tilde{\theta}(x)}\big(\tilde{\theta}(x)\big)$'; 
% cbar.Label.String = '$\mathrm{p}_{\tilde{\theta}}\big(\tilde{\theta}\big)$'; 
% cbar.Label.String = '$H(\mathrm{y};\mathrm{x})$'; 
% cbar.Label.String = '$\big\|\tilde{\theta}(x)\big\|_{\infty}$'; 


caxis(clims);   
caxis([min(PDF_dir),max(PDF_dir)])


set(gca,'Color',0.5*[1,1,1])


end

return

% figure(11); clf;
% % plot3(Theta(1,:),Theta(2,:),PDF_dir); 
% scatter3(Theta(1,:),Theta(2,:),PDF_dir,100,PDF_dir,'.');
% grid on;



% E2 = (Y.^2)'*Theta;
% E1 = Y'*Theta;
% V = sum(PDF_dir.*(E2-E1.^2)) / (N_t^2)
% 
% ii = find(~isinf(PDF_dir));
% V = sum(PDF_dir(ii).*(E2(ii)-E1(ii).^2)) / (N_t^2)
% 
% (Y.^2)'*P_y - (Y'*P_y)^2
% 
% 
% alpha_0/(alpha_0+1)*(Y.^2)'*P_y + 1/(alpha_0+1)*(Y'*P_y)^2


return






%%%
Pri = factorial(M-1)*ones(1,L_set);

Int_Pri = sum(Pri) / (N_t^2)


%%%
N_bar = [1,1,4].';

N = sum(N_bar);

Post = factorial(N+M-1) * prod(Theta.^(N_bar*ones(1,L_set)) ./ (factorial(N_bar)*ones(1,L_set)) , 1);

% Int_Post = sum(Post) / (N_t^2)


N_bar_2 = [5,5,20].';

N_2 = sum(N_bar_2);

Post_2 = factorial(N_2+M-1) * prod(Theta.^(N_bar_2*ones(1,L_set)) ./ (factorial(N_bar_2)*ones(1,L_set)) , 1);





%%% Plots 
colormap('default');

% figure(1); clf;
% subplot(2,1,1);
% scatter(Theta(1,:),Theta(2,:),100,Pri,'.');
% xlabel('$\bar{\theta}_1$','Interpreter','latex'); 
% ylabel('$\bar{\theta}_2$','Interpreter','latex'); 
% title('P$\left(\bar{\theta}\right)$','Interpreter','latex');
% colorbar;
% grid on; set(gca,'XLim',[0,1],'YLim',[0,1],'DataAspectRatio',[1,1,1]); 
% subplot(2,1,2);
% scatter3(Theta(1,:),Theta(2,:),Theta(3,:),100,Pri,'.');
% xlabel('$\theta_1$','Interpreter','latex'); 
% ylabel('$\theta_2$','Interpreter','latex'); 
% zlabel('$\theta_3$','Interpreter','latex'); 
% title('P$(\theta)$','Interpreter','latex');
% colorbar;
% grid on; set(gca,'XLim',[0,1],'YLim',[0,1],'ZLim',[0,1],'DataAspectRatio',[1,1,1]); 
% view(135,45); %view(80,5);
% 
% 
% figure(2); clf;
% subplot(2,1,1);
% scatter3(Theta(1,:),Theta(2,:),Theta(3,:),100,Post,'.');
% xlabel('$\theta_1$','Interpreter','latex'); 
% ylabel('$\theta_2$','Interpreter','latex'); 
% zlabel('$\theta_3$','Interpreter','latex'); 
% title(['P$(\theta|D)$, $\bar{N}(D)=[1,1,4]^T$'],'Interpreter','latex');
% grid on; caxis([0,max(Post)]); colorbar; axis equal;
% view(135,45); %view(80,5);
% subplot(2,1,2);
% scatter3(Theta(1,:),Theta(2,:),Theta(3,:),100,Post_2,'.');
% xlabel('$\theta_1$','Interpreter','latex'); 
% ylabel('$\theta_2$','Interpreter','latex'); 
% zlabel('$\theta_3$','Interpreter','latex'); 
% title(['P$(\theta|D)$, $\bar{N}(D)=[5,5,20]^T$'],'Interpreter','latex');
% grid on; caxis([0,max(Post_2)]); colorbar; axis equal;
% view(135,45); %view(80,5);



% figure(1); clf;
% subplot(2,1,1);
% scatter3(Theta(1,:),Theta(2,:),Theta(3,:),100,Pri,'.');
% xlabel('\theta_1'); ylabel('\theta_2'); zlabel('\theta_3');
% grid on; title('P(\theta)'); view(80,5); caxis([0,max(Post)]); colorbar; axis equal;
% subplot(2,1,2);
% scatter3(Theta(1,:),Theta(2,:),Theta(3,:),100,Post,'.');
% xlabel('\theta_1'); ylabel('\theta_2'); zlabel('\theta_3');
% grid on; title('P(\theta|D)'); view(80,5); caxis([0,max(Post)]); colorbar; axis equal;


% figure(2); clf;
% plot3(Theta(1,:),Theta(2,:),Pri,'b',Theta(1,:),Theta(2,:),Post,'r');
% xlabel('\theta_1'); ylabel('\theta_2'); legend('P(\theta)','P(\theta|D)');
% grid on; title(['Prior/Posterior PMF, N_{bar} = [',num2str(N_bar'),']^T']);



% %%% P(y|D)
% P_y_D_1 = (N_bar + 1)/(N+M);
% P_y_D_2 = (N_bar_2 + 1)/(N_2+M);
% 
% 
% figure(31); clf;
% stem(1:M,P_y_D_1)
% hold on; stem(1:M,P_y_D_2)
% grid on;




%%%%%%%%%%%%%%%%%%%%%%%%


Risk_01_opt = 1 - max(Theta);

figure(3); clf;
scatter3(Theta(1,:),Theta(2,:),Theta(3,:),100,Risk_01_opt,'.');
xlabel('\theta_1'); ylabel('\theta_2'); zlabel('\theta_3');
grid on; title('Optimal 0-1 Risk'); view(80,5); caxis([0,1-1/M]); axis equal; view(135,45);
cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}(f)$';


y = (1:3)'/3;
Risk_SE_opt = (y.^2)'*Theta - (y'*Theta).^2;

figure(4); clf;
scatter3(Theta(1,:),Theta(2,:),Theta(3,:),100,Risk_SE_opt,'.');
xlabel('\theta_1'); ylabel('\theta_2'); zlabel('\theta_3');
% xlabel('\alpha_1/\alpha_0'); ylabel('\alpha_2/\alpha_0'); zlabel('\alpha_3/\alpha_0');
grid on; view(80,5); axis equal; view(135,45); title('Optimal SE Risk'); %caxis([0,1-1/M]); 
cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}(f)$';
% cbar.Label.String = '$\Sigma_y$'; title('$\mathcal{R}(f)$, $N=0$','Interpreter','latex');



% %%% Entropy
% E = -1*sum(Theta.*log(Theta),1);
% 
% figure(5); clf;
% scatter3(Theta(1,:),Theta(2,:),Theta(3,:),100,E,'.');
% xlabel('\theta_1'); ylabel('\theta_2'); zlabel('\theta_3');
% grid on; title('Entropy'); view(80,5); caxis([0,log(M)]); colorbar; axis equal; view(135,45);


% figure(6); clf;
% plot(E,Risk_01_opt,'b.');
% grid on;







return
%%% P(D)
M = 3;
N = 3;

D = (1:M);

for n = 2:N
    temp = kron(D,ones(1,M));
    D = [repmat(1:M,[1,M^(n-1)]) ; temp];
end

N_bar = zeros(M,size(D,2));
for n = 1:N
    for idx = 1:(M^N)
        N_bar(D(n,idx),idx) = N_bar(D(n,idx),idx) + 1;
    end
end

P_D = nchoosek(N+M-1,M-1)^-1 / factorial(N) * prod(factorial(N_bar),1);

figure(3); clf;
% scatter(D(1,:),D(2,:),100,P_D,'filled');
scatter3(D(1,:),D(2,:),D(3,:),100,P_D,'filled'); 
grid on; colormap('jet');


%%% P(N|t)
M = 3;
N = 10;

theta = [0.2, 0.2, 0.6].';

N_bar = N_bar_set(M,N);
L_set = size(N_bar,2);

P = factorial(N) ./ prod(factorial(N_bar),1) .* prod((theta*ones(1,L_set)).^N_bar,1);


figure(4); clf;
scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),100,P,'filled');
grid on;



