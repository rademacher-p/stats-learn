%%%%%

clear;


M = 3;

N_t = 121;

clims = [0,0.01];

m_size = [200];


p_mm = 0.8;


Theta = N_bar_set(M,N_t)/N_t;

ii = find((sum(Theta==0) > 0) & (max(Theta,[],1) >= p_mm));
Theta = Theta(:,ii);

L_set = size(Theta,2);


   
% PDF_dir = factorial(M-2)/M * ones(1,L_set);
PDF_dir = N_t/L_set * ones(1,L_set);

sum(PDF_dir) / (N_t)

Theta_plot = Theta;



figure(10); clf;
% scatter3(Theta_plot(1,:),Theta_plot(2,:),Theta_plot(3,:),m_size(idx_vec),PDF_dir,'.');
scatter3(Theta_plot(1,:),Theta_plot(2,:),Theta_plot(3,:),m_size,PDF_dir,'.');

grid on; 
axis equal; 
view(135,45); 

xlabel('$\tilde{\theta}(\mathcal{Y}_1;x)$','Interpreter','latex'); 
ylabel('$\tilde{\theta}(\mathcal{Y}_2;x)$','Interpreter','latex');
zlabel('$\tilde{\theta}(\mathcal{Y}_3;x)$','Interpreter','latex');
% xlabel('$\theta''(\mathcal{X}_1)$','Interpreter','latex'); 
% ylabel('$\theta''(\mathcal{X}_2)$','Interpreter','latex'); 
% zlabel('$\theta''(\mathcal{X}_3)$','Interpreter','latex'); 

% title(['$\alpha(\cdot,x) = [',str_a(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');



cbar = colorbar; cbar.Label.Interpreter = 'latex'; 
cbar.Label.String = '$\mathrm{p}_{\tilde{\theta}}\big(\tilde{\theta}\big)$'; 
% cbar.Label.String = '$\mathrm{p}_{\theta''}\big(\theta''\big)$'; 

% caxis(clims);   
% caxis([min(PDF_dir),max(PDF_dir)])


% set(gca,'Color',0.5*[1,1,1])


return




%%%

clear;


M = 3;

N_t = 101;

m_size = [200];


temp = rand;
v1 = [temp; 1-temp; 0];
v2 = [0;0;1];

cvx= linspace(0,1,N_t);
Theta = v1*cvx + v2*(1-cvx);


L_set = size(Theta,2);


PDF_dir = factorial(M-2) * ones(1,L_set);
% PDF_dir = ones(1,L_set);

sum(PDF_dir) / (N_t)



Theta_plot = Theta;

figure(10); clf;
scatter3(Theta_plot(1,:),Theta_plot(2,:),Theta_plot(3,:),m_size,PDF_dir,'.');

grid on; 
axis equal; 
axis([0,1,0,1,0,1]);
view(135,45); 

xlabel('$\theta''(\mathcal{X}_1)$','Interpreter','latex'); 
ylabel('$\theta''(\mathcal{X}_2)$','Interpreter','latex'); 
zlabel('$\theta''(\mathcal{X}_3)$','Interpreter','latex'); 

cbar = colorbar; cbar.Label.Interpreter = 'latex'; 
cbar.Label.String = '$\mathrm{p}_{\theta''}\big(\theta''\big)$'; 

% caxis(clims);   
% caxis([min(PDF_dir),max(PDF_dir)])

% set(gca,'Color',0.5*[1,1,1])


