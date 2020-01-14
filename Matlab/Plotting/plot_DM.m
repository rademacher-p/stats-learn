%%%%%

clear;

N = 10;

% alpha_0_vec = [2.99,600];
alpha_0_vec = [3];


P_y = [1/3; 1/3; 1/3];
% P_y = [.3; .3; .4];


N_bar = N_bar_set(numel(P_y),N);
L_set = size(N_bar,2);



figure(1); clf;
for idx_vec = 1:numel(alpha_0_vec)
    
alpha_0 = alpha_0_vec(idx_vec);
    
alpha = alpha_0*P_y;

PMF_DM = zeros(1,L_set);
for idx = 1:L_set
    PMF_DM(idx) = coef_multi(N_bar(:,idx)) * ...
        beta_multi(alpha)^-1 * beta_multi(alpha + N_bar(:,idx));
end

sum(PMF_DM)

ii = find(isinf(PMF_DM))
PMF_DM(ii) = [];


% PMF_DM = PMF_DM * N^2; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


subplot(numel(alpha_0_vec),1,idx_vec)
scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),1000,PMF_DM,'.');
xlabel('$\bar{\mathrm{n}}(\mathcal{Y}_1,\mathcal{X}_1)$','Interpreter','latex'); 
ylabel('$\bar{\mathrm{n}}(\mathcal{Y}_2,\mathcal{X}_1)$','Interpreter','latex'); 
zlabel('$\bar{\mathrm{n}}(\mathcal{Y}_3,\mathcal{X}_1)$','Interpreter','latex'); 
vec_str_P = num2str(P_y','%0.2f,');
vec_str_alpha = num2str(alpha_0*P_y','%0.0f,');
% title(['$\mathrm{P}_{\mathrm{y},\mathrm{x}} = [',vec_str_P(1:end-1),...
%     ']^T$, $\alpha_0 = ',num2str(alpha_0),'$'],'Interpreter','latex');
title(['$\alpha = [',vec_str_alpha(1:end-1),...
    ']^T$'],'Interpreter','latex');
grid on; 

cbar = colorbar; cbar.Label.Interpreter = 'latex'; 
cbar.Label.String = '$\mathrm{P}_{\bar{\mathrm{n}}}(\bar{n}))$'; 
% yt = get(cbar,'XTick');
% set(cbar,'XTickLabel',sprintf('%0.2f',yt));

axis equal; %caxis([min(PDF_dir),max(PDF_dir)])
view(135,45); 

end





return

%%%%%

clear;

N_vec = [10, 100];

markersize = [1000,100];

alpha_0 = 6;

P_y = [1/3; 1/3; 1/3];
% P_y = [.3; .3; .4];






figure(1); clf;
for idx_vec = 1:numel(N_vec)
    
N = N_vec(idx_vec);
    
N_bar = N_bar_set(numel(P_y),N);
L_set = size(N_bar,2);
        
alpha = alpha_0*P_y;

PMF_DM = zeros(1,L_set);
for idx = 1:L_set
    PMF_DM(idx) = coef_multi(N_bar(:,idx)) * ...
        beta_multi(alpha)^-1 * beta_multi(alpha + N_bar(:,idx));
end

sum(PMF_DM)

ii = find(isinf(PMF_DM))
PMF_DM(ii) = [];


% PMF_DM = PMF_DM * N^2; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


subplot(numel(N_vec),1,idx_vec)
scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),markersize(idx_vec),PMF_DM,'.');
xlabel('$\bar{\mathrm{n}}(\mathcal{Y}_1,\mathcal{X}_1)$','Interpreter','latex'); 
ylabel('$\bar{\mathrm{n}}(\mathcal{Y}_2,\mathcal{X}_1)$','Interpreter','latex'); 
zlabel('$\bar{\mathrm{n}}(\mathcal{Y}_3,\mathcal{X}_1)$','Interpreter','latex'); 
vec_str_P = num2str(P_y','%0.2f,');
title(['$\mathrm{P}_{\mathrm{y},\mathrm{x}} = [',vec_str_P(1:end-1),...
    ']^T$, $\alpha_0 = ',num2str(alpha_0),'$'],'Interpreter','latex');
grid on; 
cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = 'p$(\bar{\mathrm{n}})$'; 
axis equal; %caxis([min(PDF_dir),max(PDF_dir)])
view(135,45); 

end


return




theta = P_y;

    
PMF_mult = zeros(1,L_set);
for idx = 1:L_set
    PMF_mult(idx) = coef_multi(N_bar(:,idx)) * prod(theta.^N_bar(:,idx));   
end


PMF_mult = PMF_mult * N^2; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


sum(PMF_mult)

figure(2); clf;
scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),1000,PMF_mult,'.');
xlabel('$\bar{\mathrm{n}}(y_1)$','Interpreter','latex'); 
ylabel('$\bar{\mathrm{n}}(y_2)$','Interpreter','latex'); 
zlabel('$\bar{\mathrm{n}}(y_3)$','Interpreter','latex'); 
vec_str_P = num2str(theta','%0.2f,');
title(['$\theta = [',vec_str_P(1:end-1),']^T$'],'Interpreter','latex');
grid on; 
cbar = colorbar; cbar.Label.Interpreter = 'latex'; 
cbar.Label.String = 'p$(\bar{\mathrm{n}} | \theta)$'; 
axis equal; %caxis([min(PDF_dir),max(PDF_dir)])
view(135,45); 


figure(3); clf;
scatter3(N_bar(1,:),N_bar(2,:),N_bar(3,:),1000,(PMF_DM-PMF_mult),'.');
xlabel('$\bar{\mathrm{n}}(y_1)$','Interpreter','latex'); 
ylabel('$\bar{\mathrm{n}}(y_2)$','Interpreter','latex'); 
zlabel('$\bar{\mathrm{n}}(y_3)$','Interpreter','latex'); 
vec_str_P = num2str(theta','%0.2f,');
title(['$\theta = [',vec_str_P(1:end-1),']^T$'],'Interpreter','latex');
grid on; 
cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = 'p$(\bar{\mathrm{n}})$'; 
axis equal; %caxis([min(PDF_dir),max(PDF_dir)])
view(135,45); 


