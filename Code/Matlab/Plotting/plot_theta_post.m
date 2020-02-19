%%%%%

clear;

alpha_0 = 10;

% alpha = [1/3; 1/3; 1/3];
alpha = [.5; .3; .2];

N_plot = 100;

Theta = N_bar_set(numel(alpha),N_plot)/N_plot;
L_set = size(Theta,2);


N_bar = [1; 1; 4];

N = sum(N_bar(:));
psi = N_bar / N;


    
    
PDF_pri = beta_multi(alpha_0*alpha)^-1 * prod(Theta.^((alpha_0*alpha-1)*ones(1,L_set)));

alpha_post = alpha_0*alpha + N*psi;
PDF_post = beta_multi(alpha_post)^-1 * prod(Theta.^((alpha_post-1)*ones(1,L_set)));



figure(1); clf;

ii = find(isinf(PDF_pri))
% Theta(:,ii);
Theta_plot = Theta;
Theta_plot(:,ii) = [];
PDF_pri(ii) = [];

% subplot(2,1,1)
subplot(1,2,1)
scatter3(Theta_plot(1,:),Theta_plot(2,:),Theta_plot(3,:),100,PDF_pri,'.');
% xlabel('$\theta(\mathcal{Y}_1,\mathcal{X}_1)$','Interpreter','latex'); 
% ylabel('$\theta(\mathcal{Y}_2,\mathcal{X}_1)$','Interpreter','latex'); 
% zlabel('$\theta(\mathcal{Y}_3,\mathcal{X}_1)$','Interpreter','latex'); 
xlabel('$\theta_{\mathrm{c}}(\mathcal{Y}_1;x)$','Interpreter','latex'); 
ylabel('$\theta_{\mathrm{c}}(\mathcal{Y}_2;x)$','Interpreter','latex');
zlabel('$\theta_{\mathrm{c}}(\mathcal{Y}_3;x)$','Interpreter','latex');

% vec_str_P = num2str(alpha','%0.0f,');
str_a = num2str(alpha','%g,');
% title(['Prior: $\alpha = [',vec_str_P(1:end-1),']^\mathrm{T}$'],'Interpreter','latex');
% title(['Prior: $\alpha(\cdot,x) = [',str_a(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');% title(['Prior: P$(y,x) = [',vec_str_P(1:end-1),']^T$, $\alpha_0 = ',num2str(alpha_0),'$'],'Interpreter','latex');
title(['Prior: $\alpha_{\mathrm{c}}(x) = [',str_a(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');% title(['Prior: P$(y,x) = [',vec_str_P(1:end-1),']^T$, $\alpha_0 = ',num2str(alpha_0),'$'],'Interpreter','latex');


grid on; 
cbar = colorbar; cbar.Label.Interpreter = 'latex'; set(gca,'CLim',[0,13]);
% cbar.Label.String = '$\mathrm{p}_{\theta}(\theta)$'; 
cbar.Label.String = '$\mathrm{p}_{\theta_{\mathrm{c}}(x)}\big(\theta_{\mathrm{c}}(x)\big)$'; 

axis equal; %caxis([min(PDF_dir),max(PDF_dir)])
view(135,45); 

ii = find(isinf(PDF_post))
% Theta(:,ii);
Theta_plot = Theta;
Theta_plot(:,ii) = [];
PDF_post(ii) = [];

% subplot(2,1,2)
subplot(1,2,2)
scatter3(Theta_plot(1,:),Theta_plot(2,:),Theta_plot(3,:),100,PDF_post,'.');
% xlabel('$\theta(\mathcal{Y}_1,\mathcal{X}_1)$','Interpreter','latex'); 
% ylabel('$\theta(\mathcal{Y}_2,\mathcal{X}_1)$','Interpreter','latex'); 
% zlabel('$\theta(\mathcal{Y}_3,\mathcal{X}_1)$','Interpreter','latex'); 
xlabel('$\theta_{\mathrm{c}}(\mathcal{Y}_1;x)$','Interpreter','latex'); 
ylabel('$\theta_{\mathrm{c}}(\mathcal{Y}_2;x)$','Interpreter','latex');
zlabel('$\theta_{\mathrm{c}}(\mathcal{Y}_3;x)$','Interpreter','latex');

vec_str_nbar = num2str(psi','%.2g,');
% title(['Posterior: $\bar{n} = [',vec_str_nbar(1:end-1),']^\mathrm{T}$'],'Interpreter','latex');
title(['Posterior: $\psi_{\mathrm{c}}(x) = [',vec_str_nbar(1:end-1),']^\mathrm{T}$'],'Interpreter','latex');
grid on; 

cbar = colorbar; cbar.Label.Interpreter = 'latex'; set(gca,'CLim',[0,13]);
% cbar.Label.String = 'p$(\theta | \mathrm{D})$'; 
cbar.Label.String = '$\mathrm{p}_{\theta | \bar{\mathrm{n}}}(\theta | \bar{n})$'; 
cbar.Label.String = '$\mathrm{p}_{\theta_{\mathrm{c}}(x) | \psi}\big(\theta_{\mathrm{c}}(x) | \psi \big)$'; 

axis equal; %caxis([min(PDF_dir),max(PDF_dir)])
view(135,45); 



return

%%%%%

clear;

alpha_0 = 10;
P_x = [0.5; 0.5];
P_y_x = [1/2; 1/2]*ones(1,numel(P_x));


N_plot = 10;

[M_y,M_x] = size(P_y_x);

Theta = N_bar_set_gen([M_y,M_x],N_plot)/N_plot;
L_set = size(Theta,3);

   
    
alpha = alpha_0*(ones(M_y,1)*P_x').*P_y_x;

PDF_pri = beta_multi(alpha)^-1 * prod(Theta.^((alpha-1)*ones(1,L_set)));


ii = find(isinf(PDF_pri))
% Theta(:,ii);
Theta_plot = Theta;
Theta_plot(:,ii) = [];
PDF_pri(ii) = [];

figure(1); clf;
scatter3(Theta_plot(1,:),Theta_plot(2,:),Theta_plot(3,:),100,PDF_pri,'.');
xlabel('$\theta(y_1)$','Interpreter','latex'); 
ylabel('$\theta(y_2)$','Interpreter','latex'); 
zlabel('$\theta(y_3)$','Interpreter','latex'); 
vec_str_P = num2str(alpha','%0.0f,');
title(['Prior: $\alpha(y,x) = [',vec_str_P(1:end-1),']^T$'],'Interpreter','latex');
% title(['Prior: P$(y,x) = [',vec_str_P(1:end-1),']^T$, $\alpha_0 = ',num2str(alpha_0),'$'],'Interpreter','latex');
grid on; 
cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = 'p$(\theta)$'; 
axis equal; %caxis([min(PDF_dir),max(PDF_dir)])
view(135,45); 
