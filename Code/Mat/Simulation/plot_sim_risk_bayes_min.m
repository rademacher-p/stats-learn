%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Minimum Bayes Risk, Dirichlet Prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

%%% Inputs

en_emp = 1;
N_mc = 1000;                  % Number of monte carlo iterations


% Y = num2cell((1:2)'/2);             % output set
% X = num2cell((1:1)'/1);             % input set

% Y = {'a';'b';'c';'d'};
Y = num2cell((1:2)');
X = num2cell((1:2)');

fcn_prior = @dirrnd;



% fcn_loss = @loss_SE;
% fcn_learn = @learn_dir_SE;
% fcn_risk_a = @risk_min_dir_SE;

fcn_loss = @loss_01;
fcn_learn = @learn_dir_01;
fcn_risk_a = @risk_min_dir_01;




% N = 1;
% N = [0, 1, 10, 100]';
% N = [0, 1, 2, 4]';
% N = [0, 2, 4, 8]';
N = (0:1:5)';

% alpha_0 = .001*numel(Y)*numel(X);
% alpha_0 = 10;
alpha_0 = numel(Y)*numel(X)*[.1, 1, 10]';
% alpha_0 = (.1:.1:20)';

P_x = ones(numel(X),1)/numel(X);
% P_x = [.99; .01];
% P_x = [.5,.5; .7,.3; .9,.1]';
% P_x = N_bar_set(numel(X),100)/100;

P_y_x = ones(numel(Y),numel(X))/numel(Y);
% P_y_x = [.8; .2]*ones(1,numel(X));
% P_y_x = [.3,.7; .5,.5; .1,.9]';
% P_y_x = repmat(permute(N_bar_set(numel(Y),100)/100,[1,3,2]),[1,numel(X)]);





%%% Iterate
L_N = numel(N);
L_alpha_0 = numel(alpha_0);
L_P_x = size(P_x,2);
L_P_y_x = size(P_y_x,3);

R_e = NaN(L_N,L_alpha_0,L_P_x,L_P_y_x);
R_a = NaN(L_N,L_alpha_0,L_P_x,L_P_y_x);

L_iter = L_N*L_alpha_0*L_P_x*L_P_y_x;
iter = 0;
for idx_alpha_0 = 1:L_alpha_0
%     fprintf(' alpha_0: %i/%i \n',idx_alpha_0,L_alpha_0); 

for idx_P_x = 1:L_P_x 
%     fprintf('  P_x: %i/%i \n',idx_P_x,L_P_x); 

for idx_P_y_x = 1:L_P_y_x
%     fprintf('   P_y_x: %i/%i \n',idx_P_y_x,L_P_y_x); 


alpha_p = P_y_x(:,:,idx_P_y_x).*(ones(numel(Y),1)*P_x(:,idx_P_x)')*alpha_0(idx_alpha_0);

fcn_learn_mc = @(x,D)fcn_learn(x,D,Y,X,alpha_p);
fcn_risk_a_mc = @(Y,X,N)fcn_risk_a(Y,X,N,alpha_p);


for idx_N = 1:L_N   
%     fprintf('N: %i/%i \n',idx_N,L_N);   


iter = iter + 1;
fprintf('Iteration: %i/%i ... \n',iter,L_iter)

N_p = N(idx_N);

if en_emp
    fcn_prior_mc = @(N_mc)fcn_prior(alpha_p,N_mc);
% %     fcn_prior_mc = fcn_prior;
           
    R_e(idx_N,idx_alpha_0,idx_P_x,idx_P_y_x) = ...
        fcn_sim_risk(N_mc,Y,X,N_p,fcn_prior_mc,fcn_learn_mc,fcn_loss);
end

R_a(idx_N,idx_alpha_0,idx_P_x,idx_P_y_x) = fcn_risk_a_mc(Y,X,N_p);


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



if (L_N == 1) && (L_alpha_0 == 1) && (L_P_x == 1) && (L_P_y_x > 1)
    
if numel(Y) == 3
    figure(1); clf;
    if en_emp
        subplot(2,1,1);
    end
    
    scatter3(P_y_x(1,1,:),P_y_x(2,1,:),P_y_x(3,1,:),100,squeeze(R_a),'.');
    xlabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_1)$','Interpreter','latex'); 
    ylabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_2)$','Interpreter','latex');
    zlabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_3)$','Interpreter','latex');
    grid on; axis equal; view(135,45);
    vec_str_x = num2str(P_x','%0.1f,');
    title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$',...
        ', $\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
%     title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$',...
%         ', P$(x) = [',vec_str_x(1:end-1),']^T$'],'Interpreter','latex');
    cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}^*(f)$';

    if en_emp
        subplot(2,1,2);
        scatter3(P_y_x(1,1,:),P_y_x(2,1,:),P_y_x(3,1,:),100,squeeze(R_e),'.');
        xlabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_1)$','Interpreter','latex'); 
        ylabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_2)$','Interpreter','latex');
        zlabel('$\mathrm{P}_{\mathrm{y}|\mathrm{x}}(\mathcal{Y}_3)$','Interpreter','latex');
        grid on; axis equal; view(135,45);
        vec_str_x = num2str(P_x','%0.1f,');
        title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$',...
            ', $\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
    %     title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$',...
    %         ', P$(x) = [',vec_str_x(1:end-1),']^T$'],'Interpreter','latex');
        cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}^*(f)$';
    end
    
%     subplot(2,1,2);
%     scatter3(P_x(1,:),P_x(2,:),P_x(3,:),350,squeeze(R_e),'.');
%     xlabel('$\alpha''(x_1)/\alpha_0$','Interpreter','latex'); 
%     ylabel('$\alpha''(x_2)/\alpha_0$','Interpreter','latex'); 
%     zlabel('$\alpha''(x_3)/\alpha_0$','Interpreter','latex'); 
%     grid on; axis equal; view(135,45);
%     title([title_loss,'Risk (sim), $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$'],...
%         'Interpreter','latex');
%     cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}^*(f)$';
end

end




if (L_N == 1) && (L_alpha_0 == 1) && (L_P_x > 1) && (L_P_y_x == 1)
    
if numel(X) == 3
    figure(1); clf;
%     subplot(2,1,1);
    scatter3(P_x(1,:),P_x(2,:),P_x(3,:),100,squeeze(R_a),'.');
    xlabel('$\mathrm{P}_{\mathrm{x}}(\mathcal{X}_1)$','Interpreter','latex'); 
    ylabel('$\mathrm{P}_{\mathrm{x}}(\mathcal{X}_2)$','Interpreter','latex');
    zlabel('$\mathrm{P}_{\mathrm{x}}(\mathcal{X}_3)$','Interpreter','latex');
%     xlabel('$\alpha''(x_1)/\alpha_0$','Interpreter','latex'); 
%     ylabel('$\alpha''(x_2)/\alpha_0$','Interpreter','latex'); 
%     zlabel('$\alpha''(x_3)/\alpha_0$','Interpreter','latex'); 
    grid on; axis equal; view(135,45);
    vec_str_y_x = num2str(unique(P_y_x','rows'),'%0.1f,');
    title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$',...
        ', $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',vec_str_y_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
    cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}^*(f)$';
%     subplot(2,1,2);
%     scatter3(P_x(1,:),P_x(2,:),P_x(3,:),350,squeeze(R_e),'.');
%     xlabel('$\alpha''(x_1)/\alpha_0$','Interpreter','latex'); 
%     ylabel('$\alpha''(x_2)/\alpha_0$','Interpreter','latex'); 
%     zlabel('$\alpha''(x_3)/\alpha_0$','Interpreter','latex'); 
%     grid on; axis equal; view(135,45);
%     title([title_loss,'Risk (sim), $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$'],...
%         'Interpreter','latex');
%     cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}^*(f)$';
end

end




%%% Lines versus N or alpha_0, other for legend
if ((L_N > 1) || (L_alpha_0 > 1)) && (L_P_x == 1) && (L_P_y_x == 1)

    if L_N >= L_alpha_0
    figure(1); clf;
    sub_leg = [];
    str_leg = cell(1,L_alpha_0);
    for idx = 1:L_alpha_0
        hold on;
        p_a = plot(N,R_a(:,idx,1,1),'.-','MarkerSize',15,'Color',colors(1+mod(idx-1,N_clr),:));
        p_e = plot(N,R_e(:,idx,1,1),'o-','Color',colors(1+mod(idx-1,N_clr),:)); 

        sub_leg = [sub_leg, p_a];    
        str_leg(idx) = {['$\alpha_0 = ',num2str(alpha_0(idx)),'$']};
    end
    grid on; %set(gca,'YLim',[0,1]);
    vec_str_y_x = num2str(unique(P_y_x','rows'),'%0.1f,');
    vec_str_x = num2str(P_x','%0.1f,'); %vec_str_x = num2str(P_x','%0.1f,'); 
    title([title_loss,' Risk, $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',vec_str_y_x(1:end-1),']^{\mathrm{T}}$',...
        ', $\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');    
    xlabel('$N$','Interpreter','latex'); 
    ylabel('$\mathcal{R}^*(f)$','Interpreter','latex'); 
    legend(sub_leg,str_leg,'Interpreter','latex','Location','northeast');
    end

    if L_N < L_alpha_0
    figure(1); clf;
    sub_leg = [];
    str_leg = cell(1,L_N);
    for idx = 1:L_N
        hold on;
        p_a = plot(alpha_0,R_a(idx,:,1,1)','-','LineWidth',2,'Color',colors(1+mod(idx-1,N_clr),:));
        p_e = plot(alpha_0,R_e(idx,:,1,1)','--','LineWidth',2,'Color',colors(1+mod(idx-1,N_clr),:)); 

        sub_leg = [sub_leg, p_a];    
        str_leg(idx) = {['$N = ',num2str(N(idx)),'$']};
    end
    grid on; %set(gca,'YLim',[0,1]);
    vec_str_y_x = num2str(unique(P_y_x','rows'),'%0.1f,');
    vec_str_x = num2str(P_x','%0.1f,'); 
    title([title_loss,' Risk, $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',vec_str_y_x(1:end-1),']^{\mathrm{T}}$',...
        ', $\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
    xlabel('$\alpha_0$','Interpreter','latex'); 
    ylabel('$\mathcal{R}^*(f)$','Interpreter','latex'); 
    legend(sub_leg,str_leg,'Interpreter','latex','Location','southeast');
    end

end


if (L_N > 1) && (L_alpha_0 == 1) && (L_P_x > 1) && (L_P_y_x == 1)
    
figure(1); clf;
sub_leg = [];
str_leg = cell(1,L_P_x);
for idx = 1:L_P_x
    hold on;
    p_a = plot(N,R_a(:,1,idx,1),'.','Color',colors(1+mod(idx-1,N_clr),:));
    p_e = plot(N,R_e(:,1,idx,1),'o','Color',colors(1+mod(idx-1,N_clr),:)); 

    sub_leg = [sub_leg, p_a]; 
    vec_str_x = num2str(P_x(:,idx)','%0.1f,');
    str_leg(idx) = {['$\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$']};
end
grid on; %set(gca,'YLim',[0,1]);\
vec_str_y_x = num2str(unique(P_y_x','rows'),'%0.1f,');
title([title_loss,' Risk, $\alpha_0 = ',num2str(alpha_0),...
    '$, $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',...
    vec_str_y_x(1:end-1),']^{\mathrm{T}}$'],'Interpreter','latex');
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*(f)$','Interpreter','latex'); 
legend(sub_leg,str_leg,'Interpreter','latex','Location','northeast');

end


if (L_N == 1) && (L_alpha_0 > 1) && (L_P_x > 1) && (L_P_y_x == 1)
    
figure(1); clf;
sub_leg = [];
str_leg = cell(1,L_P_x);
for idx = 1:L_P_x
    hold on;
    p_a = plot(alpha_0,R_a(1,:,idx,1),'-','Color',colors(1+mod(idx-1,N_clr),:));
    p_e = plot(alpha_0,R_e(1,:,idx,1),'--','Color',colors(1+mod(idx-1,N_clr),:)); 

    sub_leg = [sub_leg, p_a]; 
    vec_str_x = num2str(P_x(:,idx)','%0.1f,');
    str_leg(idx) = {['$\mathrm{P}_{\mathrm{x}} = [',vec_str_x(1:end-1),']^{\mathrm{T}}$']};
end
grid on; %set(gca,'YLim',[0,1]);
vec_str_y_x = num2str(unique(P_y_x','rows'),'%0.1f,');
title([title_loss,' Risk, $N = ',num2str(N),'$, $\mathrm{P}_{\mathrm{y}|\mathrm{x}} = [',...
    vec_str_y_x(1:end-1),']^\{\mathrm{T}}$'],'Interpreter','latex');
xlabel('$\alpha_0$','Interpreter','latex'); 
ylabel('$\mathcal{R}^*(f)$','Interpreter','latex'); 
legend(sub_leg,str_leg,'Interpreter','latex','Location','southeast');

end



