%%%%%%%%%% Simulation: General Model

clear;

%%% Inputs

fcn_loss = @loss_01;
fcn_learn = @learn_opt_01;
% fcn_learn_b = @learn_opt_01_basic;
fcn_risk_a = @risk_opt_01;
% Y = {'a';'b';'c';'d'};
% Y = {'a';'b'};
Y = num2cell((1:3)');
X = num2cell((1:1)');

% fcn_loss = @loss_SE;
% fcn_learn = @learn_opt_SE;
% % fcn_learn_b = @learn_opt_SE_basic; %%%%%%%%%%%%%%%
% fcn_risk_a = @risk_opt_SE;
% Y = num2cell((1:3)'/3);             % output set
% X = num2cell((1:1)');             % input set


en_mc = 1;
N_mc = 1000;                  % Number of monte carlo iterations

N_t = 40;
mu_theta = N_bar_set(numel(Y)*numel(X),N_t)/N_t;

alpha_0 = numel(Y)*numel(X);
% alpha_0 = 100;

N = 0;


%%% Iterate
R_e = NaN(1,size(mu_theta,2));
% R_a = NaN(1,size(mu_theta,2));
for idx_t = 1:size(mu_theta,2)   

    fprintf('MC Scenario %i/%i \n',idx_t,size(mu_theta,2));

    alpha = alpha_0*mu_theta(:,idx_t);

    if en_mc
        R_e(idx_t) = fcn_MC_general(Y,X,alpha,N,fcn_loss,fcn_learn,N_mc);
    end

%     R_a(idx_t) = fcn_risk_a(Y,alpha,N);

end


%%% Plots

if strcmpi(func2str(fcn_loss),'loss_01') 
    title_loss = '0-1'; 
elseif strcmpi(func2str(fcn_loss),'loss_SE') 
    title_loss = 'SE'; 
else
    title_loss = '';
end 


% figure(1); clf;
% scatter3(mu_theta(1,:),mu_theta(2,:),mu_theta(3,:),100,R_a,'.');
% xlabel('$\alpha_1/\alpha_0$','Interpreter','latex'); 
% ylabel('$\alpha_2/\alpha_0$','Interpreter','latex'); 
% zlabel('$\alpha_3/\alpha_0$','Interpreter','latex'); 
% grid on; view(80,5); axis equal; view(135,45);
% title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$'],...
%     'Interpreter','latex');
% cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}(f)$';


figure(2); clf;
scatter3(mu_theta(1,:),mu_theta(2,:),mu_theta(3,:),350,R_e,'.');
xlabel('$\alpha_1/\alpha_0$','Interpreter','latex'); 
ylabel('$\alpha_2/\alpha_0$','Interpreter','latex'); 
zlabel('$\alpha_3/\alpha_0$','Interpreter','latex'); 
grid on; axis equal; view(135,45);
title([title_loss,' Risk, $N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$'],...
    'Interpreter','latex');
cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = '$\mathcal{R}(f)$';



