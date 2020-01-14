%%%%%%%%%% Simulation: General Model

clear;

%%% Inputs

en_emp = 0;
N_mc = 2000;                  % Number of monte carlo iterations


Y = num2cell((1:3)'/3);             % output set
X = num2cell((1:1)');             % input set

% % Y = {'a';'b';'c';'d'};
% Y = num2cell((1:4)');
% X = num2cell((1:2)');


% fcn_loss = @loss_SE;
% fcn_risk_clv = @risk_clv_SE;
% fcn_learn = @learn_dir_SE;
% fcn_risk_an = @risk_cond_dir_SE;

fcn_loss = @loss_01;
fcn_risk_clv = @risk_clv_01;
fcn_learn = @learn_dir_01;
fcn_risk_an = @risk_cond_dir_01;




%%%%%%%%%% PGR: FORM ANALYTICAL FCNS, ANY COMPLEXITY!!!


theta_m = ones(numel(X),1)/numel(X);
% theta_m = [.7; .3];
% theta_m = [.5,.5; .7,.3; .9,.1]';
% theta_m = N_bar_set(numel(X),100)/100;

% theta_c = ones(numel(Y),numel(X))/numel(Y);
% theta_c = repmat([.8; .2],[1,numel(X)]);
% theta_c = repmat(cat(3,[.5;.5], [.3;.7], [.1;.9]),[1,numel(X)]);
% theta_c = repmat(cat(3,[1/3;1/3;1/3], [.6;.2;.2], [.8;.1;.1]),[1,numel(X)]);
theta_c = repmat(N_bar_set_gen([numel(Y),1],80)/80,[1,numel(X)]);


N = 1;
% N = [0, 1, 10, 100]';
% N = [0, 1, 2, 4]';
% N = [0, 2, 4, 8]';
% N = (0:1:30)';

% alpha_0 = numel(Y)*numel(X);
alpha_0 = [30];
% alpha_0 = numel(Y)*numel(X)*[.1, 1, 10]';
% alpha_0 = (.1:.1:20)';
% alpha_0 = [.1; (1:20)'];

alpha_m = ones(numel(X),1)/numel(X);
% alpha_m = [.99; .01];
% alpha_m = [.5,.5; .7,.3; .9,.1]';
% alpha_m = N_bar_set(numel(X),100)/100;

% alpha_c = ones(numel(Y),numel(X))/numel(Y);
% alpha_c = [.8; .2]*ones(1,numel(X));
% alpha_c = [.8;.1;.1]*ones(1,numel(X));
% alpha_c = cat(3,[.5;.5], [.3;.7], [.1;.9]);
alpha_c = repmat(cat(3,[1/3;1/3;1/3], [.8;.1;.1]),[1,numel(X)]);
% alpha_c = repmat(N_bar_set_gen([numel(Y),1],10)/10,[1,numel(X)]);





%%% Iterate

L_theta_m = size(theta_m,2);
L_theta_c = size(theta_c,3);
L_N = numel(N);
L_alpha_0 = numel(alpha_0);
L_alpha_m = size(alpha_m,2);
L_alpha_c = size(alpha_c,3);

R_clv = NaN(L_theta_m,L_theta_c);
R_emp = NaN(L_theta_m,L_theta_c,L_N,L_alpha_0,L_alpha_m,L_alpha_c);
R_an = NaN(L_theta_m,L_theta_c,L_N,L_alpha_0,L_alpha_m,L_alpha_c);

L_iter = L_theta_m*L_theta_c*L_N*L_alpha_0*L_alpha_m*L_alpha_c;
iter = 0;


for idx_t_m = 1:L_theta_m 
%     fprintf('  theta_m: %i/%i \n',idx_P_x,L_P_x); 

for idx_t_c = 1:L_theta_c
%     fprintf('   theta_c: %i/%i \n',idx_P_y_x,L_P_y_x); 


theta_p = theta_c(:,:,idx_t_c).*(ones(numel(Y),1)*theta_m(:,idx_t_m)');

fcn_prior = @(N_mc)repmat(theta_p,[1,1,N_mc]);

R_clv(idx_t_m,idx_t_c) = fcn_risk_clv(Y,X,theta_p);


for idx_a_0 = 1:L_alpha_0
%     fprintf(' alpha_0: %i/%i \n',idx_alpha_0,L_alpha_0); 

for idx_a_m = 1:L_alpha_m 
%     fprintf('  alpha_m: %i/%i \n',idx_P_x,L_P_x); 

for idx_a_c = 1:L_alpha_c
%     fprintf('   alpha_c: %i/%i \n',idx_P_y_x,L_P_y_x); 

alpha_p = alpha_c(:,:,idx_a_c).*(ones(numel(Y),1)*alpha_m(:,idx_a_m)')*alpha_0(idx_a_0);

fcn_learn_mc = @(x,D)fcn_learn(x,D,Y,X,alpha_p);
% fcn_risk_an_mc = @(Y,X,N)fcn_risk_an(Y,X,N,theta_p,alpha_p);


for idx_N = 1:L_N   
%     fprintf('N: %i/%i \n',idx_N,L_N);  

iter = iter + 1;
fprintf('Iteration: %i/%i ... \n',iter,L_iter)


N_p = N(idx_N);

if en_emp           
    R_emp(idx_t_m,idx_t_c,idx_N,idx_a_0,idx_a_m,idx_a_c) = ...
        fcn_MC_general(N_mc,Y,X,N_p,fcn_prior,fcn_learn_mc,fcn_loss);
else
%     R_an(idx_t_m,idx_t_c,idx_N,idx_a_0,idx_a_m,idx_a_c) = fcn_risk_an_mc(Y,X,N_p);
    R_an(idx_t_m,idx_t_c,idx_N,idx_a_0,idx_a_m,idx_a_c) = fcn_risk_an(Y,X,N_p,theta_p,alpha_p);
end



end
end
end
end
end
end






%%% Plots

colors =  [0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% colors = parula(numel(alpha_0_p));

N_clr = size(colors,1);

if strcmpi(func2str(fcn_loss),'loss_01') 
    title_loss = '0-1 '; 
elseif strcmpi(func2str(fcn_loss),'loss_SE') 
    title_loss = 'SE '; 
else
    title_loss = '';
end




if en_emp %%%%%%%%%%%%%%
    R_p = R_emp;
else
    R_p = R_an;
end




if (L_theta_m == 1) && (L_theta_c >= 1) && ...
        (L_N == 1) && (L_alpha_0 == 1) && (L_alpha_m == 1) && (L_alpha_c > 1)
    
if (numel(Y) == 3) && (numel(X) == 1)
    
figure(1); clf;
for idx = 1:L_alpha_c
    
subplot(L_alpha_c,1,idx);
scatter3(theta_c(1,1,:),theta_c(2,1,:),theta_c(3,1,:),100,squeeze(R_p(1,:,1,1,1,idx)-R_clv),'.');

xlabel('$\tilde{\theta}(\mathcal{Y}_1;x)$','Interpreter','latex'); 
ylabel('$\tilde{\theta}(\mathcal{Y}_2;x)$','Interpreter','latex');
zlabel('$\tilde{\theta}(\mathcal{Y}_3;x)$','Interpreter','latex');
grid on; axis equal; view(135,45);

str_a_c = num2str(alpha_c(:,:,idx)','%0.1f,');
title([title_loss,' Risk',...
    ', $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',str_a_c(1:end-1),']^{\mathrm{T}}$',...
    ', $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$'],'Interpreter','latex');

cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}_{\Theta}(f;\theta)$';
caxis([0,1]);    

end

end

end



if (L_theta_m == 1) && (L_theta_c >= 1) && ...
        (L_N == 1) && (L_alpha_0 >= 1) && (L_alpha_m == 1) && (L_alpha_c == 1)
    
if (numel(Y) == 3) && (numel(X) == 1)
    
figure(1); clf;
for idx = 1:L_alpha_0
    
subplot(L_alpha_0,1,idx);
scatter3(theta_c(1,1,:),theta_c(2,1,:),theta_c(3,1,:),100,squeeze(R_p(1,:,1,idx,1,1)-R_clv),'.');

xlabel('$\tilde{\theta}(\mathcal{Y}_1;x)$','Interpreter','latex'); 
ylabel('$\tilde{\theta}(\mathcal{Y}_2;x)$','Interpreter','latex');
zlabel('$\tilde{\theta}(\mathcal{Y}_3;x)$','Interpreter','latex');
grid on; axis equal; view(135,45);

str_a_0 = num2str(alpha_0(idx)','%0.1f,');
str_a_c = num2str(alpha_c','%0.1f,');

title([title_loss,' Risk',...
    ', $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',str_a_c(1:end-1),']^{\mathrm{T}}$',...
    ', $N = ',num2str(N),'$, $\alpha_0 = ',str_a_0,'$'],'Interpreter','latex');

cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}_{\Theta}(f;\theta)$';
caxis([0,1]);    

end

end

end










%%%%%
return 
%%%%%



if (L_theta_m == 1) && (L_theta_c == 1) && ...
        ((L_N > 1) || (L_alpha_0 > 1)) && (L_alpha_m == 1) && (L_alpha_c == 1)

    if L_N >= L_alpha_0
    figure(1); clf;
    sub_leg = [];
    str_leg = cell(1,L_alpha_0+1);
    
    p_min = plot([N(1),N(end)],R_clv*[1,1],'--k');
    sub_leg = [sub_leg, p_min];    
    str_leg(1) = {'$\mathcal{R}_{\Theta}^*(\theta)$'};
    for idx = 1:L_alpha_0
        hold on;
        p_a = plot(N,squeeze(R_an(1,1,:,idx,1,1)),'-.','Color',colors(1+mod(idx-1,N_clr),:));
        p_e = plot(N,squeeze(R_emp(1,1,:,idx,1,1)),'-o','Color',colors(1+mod(idx-1,N_clr),:));  

        sub_leg = [sub_leg, p_a];    
        str_leg(idx+1) = {['$\alpha_0 = ',num2str(alpha_0(idx)),'$']};
    end

    
    grid on; %set(gca,'YLim',[0,1]);
    vec_str_theta = num2str(theta_p','%0.1f,');
    vec_str_y_x = num2str(unique(alpha_c','rows'),'%0.1f,');
    vec_str_x = num2str(alpha_m','%0.1f,'); %vec_str_x = num2str(P_x','%0.1f,'); 
    title([title_loss,' Risk, '...
        '$\mathrm{P}_{\mathrm{y}|\mathrm{x},\theta} = [',vec_str_theta(1:end-1),']^{\mathrm{T}}$, ',...
        '$\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',vec_str_y_x(1:end-1),']^{\mathrm{T}}$, ',...
        '$\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');    
    xlabel('$N$','Interpreter','latex'); 
    ylabel('$\mathcal{R}_{\Theta}(f;\theta)$','Interpreter','latex'); 
    legend(sub_leg,str_leg,'Interpreter','latex','Location','northeast');
    end

    if L_N < L_alpha_0
    figure(1); clf;
    sub_leg = [];
    str_leg = cell(1,L_N+1);
    
    p_min = plot([alpha_0(1),alpha_0(end)],R_clv*[1,1],'--k');
    sub_leg = [sub_leg, p_min];    
    str_leg(1) = {['$\mathcal{R}_{\Theta}^*(\theta)$']};
    for idx = 1:L_N
        hold on;
        p_a = plot(alpha_0,R_an(1,1,idx,:,1,1)','-','Color',colors(1+mod(idx-1,N_clr),:));
        p_e = plot(alpha_0,R_emp(1,1,idx,:,1,1)','--','Color',colors(1+mod(idx-1,N_clr),:)); 
        
        sub_leg = [sub_leg, p_a];    
        str_leg(idx+1) = {['$N = ',num2str(N(idx)),'$']};
    end

    
    grid on; %set(gca,'YLim',[0,1]);
    vec_str_theta = num2str(theta_p','%0.1f,');
    vec_str_y_x = num2str(unique(alpha_c','rows'),'%0.1f,');
    vec_str_x = num2str(alpha_m','%0.1f,'); 
    title([title_loss,' Risk, '...
        '$\mathrm{P}_{\mathrm{y}|\mathrm{x},\theta} = [',vec_str_theta(1:end-1),']^{\mathrm{T}}$, ',...
        '$\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',vec_str_y_x(1:end-1),']^{\mathrm{T}}$, ',...
        '$\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');    
    xlabel('$\alpha_0$','Interpreter','latex'); 
    ylabel('$\mathcal{R}_{\Theta}(f;\theta)$','Interpreter','latex'); 
    legend(sub_leg,str_leg,'Interpreter','latex','Location','southeast');
    end

end



if (L_theta_m == 1) && (L_theta_c == 1) && ...
        (L_N == 1) && (L_alpha_0 == 1) && (L_alpha_m == 1) && (L_alpha_c > 1)
    
if numel(Y) == 3
    figure(1); clf;
    if en_emp
        subplot(2,1,1);
    end
    
    scatter3(alpha_c(1,1,:),alpha_c(2,1,:),alpha_c(3,1,:),100,squeeze(R_an),'.');
    xlabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_1|x)$','Interpreter','latex'); 
    ylabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_2|x)$','Interpreter','latex');
    zlabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_3|x)$','Interpreter','latex');
    grid on; axis equal; view(135,45);
    vec_str_theta = num2str(theta_p','%0.1f,');
    vec_str_x = num2str(alpha_m','%0.1f,');
    title([title_loss,' Risk, ',...
        '$\mathrm{P}_{\mathrm{y}|\mathrm{x},\theta} = [',vec_str_theta(1:end-1),']^{\mathrm{T}}$, ',...
        '$N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$, ',...
        '$\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
%     title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$',...
%         ', P$(x) = [',vec_str_x(1:end-1),']^T$'],'Interpreter','latex');
    cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}_{\Theta}(f;\theta)$';

    if en_emp
        subplot(2,1,2);
        scatter3(alpha_c(1,1,:),alpha_c(2,1,:),alpha_c(3,1,:),100,squeeze(R_emp),'.');
        xlabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_1|x)$','Interpreter','latex'); 
        ylabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_2|x)$','Interpreter','latex');
        zlabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_3|x)$','Interpreter','latex');
        grid on; axis equal; view(135,45);
%         vec_str_x = num2str(P_x','%0.1f,');
        title([title_loss,' Risk, ',...
            '$\mathrm{P}_{\mathrm{y}|\mathrm{x},\theta} = [',vec_str_theta(1:end-1),']^{\mathrm{T}}$, ',...
            '$N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$, ',...
            '$\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
    %     title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$',...
    %         ', P$(x) = [',vec_str_x(1:end-1),']^T$'],'Interpreter','latex');
        cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}_{\Theta}(f;\theta)$';
    end
    
%     subplot(2,1,2);
%     scatter3(P_x(1,:),P_x(2,:),P_x(3,:),350,squeeze(R_e),'.');
%     xlabel('$\alpha''(x_1)/\alpha_0$','Interpreter','latex'); 
%     ylabel('$\alpha''(x_2)/\alpha_0$','Interpreter','latex'); 
%     zlabel('$\alpha''(x_3)/\alpha_0$','Interpreter','latex'); 
%     grid on; axis equal; view(135,45);
%     title([title_loss,'Risk (sim), $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$'],...
%         'Interpreter','latex');
%     cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}(f)$';
end

end




if (L_theta_m == 1) && (L_theta_c == 1) && ...
        (L_N == 1) && (L_alpha_0 == 1) && (L_alpha_m > 1) && (L_alpha_c == 1)
    
if numel(X) == 3
    figure(1); clf;
%     subplot(2,1,1);
    scatter3(alpha_m(1,:),alpha_m(2,:),alpha_m(3,:),100,squeeze(R_an),'.');
    xlabel('$\mathrm{P}_{\mathrm{x}}(\mathcal{X}_1)$','Interpreter','latex'); 
    ylabel('$\mathrm{P}_{\mathrm{x}}(\mathcal{X}_2)$','Interpreter','latex');
    zlabel('$\mathrm{P}_{\mathrm{x}}(\mathcal{X}_3)$','Interpreter','latex');
%     xlabel('$\alpha''(x_1)/\alpha_0$','Interpreter','latex'); 
%     ylabel('$\alpha''(x_2)/\alpha_0$','Interpreter','latex'); 
%     zlabel('$\alpha''(x_3)/\alpha_0$','Interpreter','latex'); 
    grid on; axis equal; view(135,45);
    vec_str_y_x = num2str(unique(alpha_c','rows'),'%0.1f,');
    title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$',...
        ', $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',vec_str_y_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
    cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}(f)$';
%     subplot(2,1,2);
%     scatter3(P_x(1,:),P_x(2,:),P_x(3,:),350,squeeze(R_e),'.');
%     xlabel('$\alpha''(x_1)/\alpha_0$','Interpreter','latex'); 
%     ylabel('$\alpha''(x_2)/\alpha_0$','Interpreter','latex'); 
%     zlabel('$\alpha''(x_3)/\alpha_0$','Interpreter','latex'); 
%     grid on; axis equal; view(135,45);
%     title([title_loss,'Risk (sim), $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$'],...
%         'Interpreter','latex');
%     cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}(f)$';
end

end




if (L_theta_m == 1) && (L_theta_c == 1) && ... 
        (L_N > 1) && (L_alpha_0 == 1) && (L_alpha_m == 1) && (L_alpha_c > 1)
    
if L_N >= L_alpha_c
    figure(1); clf;
    sub_leg = [];
    str_leg = cell(1,1+L_alpha_c);
    
    p_min = plot([N(1),N(end)],R_clv*[1,1],'--k');
    sub_leg = [sub_leg, p_min];    
    str_leg(1) = {'$\mathcal{R}_{\Theta}^*(\theta)$'};
    
    for idx = 1:L_alpha_c
        hold on;
        p_a = plot(N,squeeze(R_an(1,1,:,1,1,idx)),'-.','Color',colors(1+mod(idx-1,N_clr),:));
        p_e = plot(N,squeeze(R_emp(1,1,:,1,1,idx)),'-o','Color',colors(1+mod(idx-1,N_clr),:));  

        sub_leg = [sub_leg, p_a];  
        
        a = alpha_0*alpha_m*alpha_c(:,:,idx);
        str_a = num2str(a','%0.1f,');
        str_leg(idx+1) = {['$\alpha = [',str_a(1:end-1),']^{\mathrm{T}}$']};
    end

    
    grid on; %set(gca,'YLim',[0,1]);
    vec_str_theta = num2str(theta_p','%0.1f,');
    title([title_loss,' Risk, '...
        '$\mathrm{P}_{\mathrm{y}|\mathrm{x},\theta} = [',vec_str_theta(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');    
    xlabel('$N$','Interpreter','latex'); 
    ylabel('$\mathcal{R}_{\Theta}(f;\theta)$','Interpreter','latex'); 
    legend(sub_leg,str_leg,'Interpreter','latex','Location','northeast');
end

end



if (L_theta_m == 1) && (L_theta_c == 1) && ...
        (L_N > 1) && (L_alpha_0 == 1) && (L_alpha_m > 1) && (L_alpha_c == 1)
    
figure(1); clf;
sub_leg = [];
str_leg = cell(1,L_alpha_m);
for idx = 1:L_alpha_m
    hold on;
    p_a = plot(N,R_an(1,1,:,1,idx,1),'.','Color',colors(1+mod(idx-1,N_clr),:));
    p_e = plot(N,R_emp(1,1,:,1,idx,1),'o','Color',colors(1+mod(idx-1,N_clr),:)); 

    sub_leg = [sub_leg, p_a]; 
    vec_str_x = num2str(alpha_m(:,idx)','%0.1f,');
    str_leg(idx) = {['$\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$']};
end
grid on; %set(gca,'YLim',[0,1]);\
vec_str_y_x = num2str(unique(alpha_c','rows'),'%0.1f,');
title([title_loss,' Risk, $\alpha_0 = ',num2str(alpha_0),...
    '$, $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',...
    vec_str_y_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
legend(sub_leg,str_leg,'Interpreter','latex','Location','northeast');

end


if (L_theta_m == 1) && (L_theta_c == 1) && ...
        (L_N == 1) && (L_alpha_0 > 1) && (L_alpha_m > 1) && (L_alpha_c == 1)
    
figure(1); clf;
sub_leg = [];
str_leg = cell(1,L_alpha_m);
for idx = 1:L_alpha_m
    hold on;
    p_a = plot(alpha_0,R_an(1,1,1,:,idx,1),'-','Color',colors(1+mod(idx-1,N_clr),:));
    p_e = plot(alpha_0,R_emp(1,1,1,:,idx,1),'--','Color',colors(1+mod(idx-1,N_clr),:)); 

    sub_leg = [sub_leg, p_a]; 
    vec_str_x = num2str(alpha_m(:,idx)','%0.1f,');
    str_leg(idx) = {['$\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$']};
end
grid on; %set(gca,'YLim',[0,1]);
vec_str_y_x = num2str(unique(alpha_c','rows'),'%0.1f,');
title([title_loss,' Risk, $N = ',num2str(N),'$, $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',...
    vec_str_y_x(1:end-1),']^\{\mathrm{T}}$'],'Interpreter','latex');
xlabel('$\alpha_0$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
legend(sub_leg,str_leg,'Interpreter','latex','Location','southeast');

end











