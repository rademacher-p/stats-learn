%%%%%%%%%% Simulation: General Model

clear;

%%% Inputs

fcn_loss = @loss_01;
fcn_learn = @learn_opt_01;
% fcn_learn_b = @learn_opt_01_basic;
fcn_risk_a = @risk_opt_01;
% Y = {'a';'b';'c';'d'};
% Y = {'a';'b'};
Y = num2cell((1:2)');
X = num2cell((1:1)');

% fcn_loss = @loss_SE;
% fcn_learn = @learn_opt_SE;
% % fcn_learn_b = @learn_opt_SE_basic; %%%%%%%%%%%%%%%
% fcn_risk_a = @risk_opt_SE;
% Y = num2cell((1:2)'/2);             % output set
% X = num2cell((1:1)');             % input set


en_mc = 1;
N_mc = 1000;                  % Number of monte carlo iterations


mu_theta = ones(numel(Y),numel(X))/numel(Y)/numel(X);
% mu_theta = [.9; .1];

% alpha_0_p = numel(Y)*numel(X);
% alpha_0_p = [.1, 1, 10]';
alpha_0_p = (0.5:0.5:20)';

% N_p = (0:25)';
% N_p = [0, 1, 100]';
N_p = 1000;



%%% Iterate

if sum(mu_theta < 0) ~= 0
    disp('Error: Invalid prior mean')
    return
elseif abs(sum(mu_theta(:)) - 1) > eps
    disp('Warning: Prior mean not normalized');
    mu_theta = mu_theta / sum(mu_theta(:));
end


% R_b = zeros(numel(N_p),numel(alpha_0_p));

R_e = NaN(numel(N_p),numel(alpha_0_p));
R_a = NaN(numel(N_p),numel(alpha_0_p));
for idx_n = 1:numel(N_p)
    for idx_a = 1:numel(alpha_0_p)   
        
        fprintf('MC Scenario %i/%i \n',(idx_n-1)*numel(alpha_0_p)+idx_a,numel(N_p)*numel(alpha_0_p));
        
        N = N_p(idx_n);
        alpha = alpha_0_p(idx_a)*mu_theta;
        
%         R_b(idx_n,idx_a) = fcn_MC_basic(Y,alpha,N,fcn_loss,fcn_learn_b,N_mc);
        
        if en_mc
            R_e(idx_n,idx_a) = fcn_MC_general(Y,X,alpha,N,fcn_loss,fcn_learn,N_mc);
        end
        
        R_a(idx_n,idx_a) = fcn_risk_a(Y,alpha,N);
        
    end
end




%%% Plots

% colors = parula(numel(alpha_0_p));
colors =  [0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];

N_clr = size(colors,1);

if strcmpi(func2str(fcn_loss),'loss_01') 
    title_loss = '0-1'; 
elseif strcmpi(func2str(fcn_loss),'loss_SE') 
    title_loss = 'SE'; 
else
    title_loss = '';
end 


figure(1); clf;
sub_leg = [];
str_leg = cell(1,numel(alpha_0_p));
for idx = 1:numel(alpha_0_p)
    hold on;
    p_a = plot(N_p,R_a(:,idx),'.','Color',colors(1+mod(idx-1,N_clr),:));
    p_e = plot(N_p,R_e(:,idx),'o','Color',colors(1+mod(idx-1,N_clr),:)); 
    
    sub_leg = [sub_leg, p_a];    
    str_leg(idx) = {['$\alpha_0 = ',num2str(alpha_0_p(idx)),'$']};
end
grid on; %set(gca,'YLim',[0,1]);
vec_str = num2str(mu_theta','%0.1f,'); title([title_loss,' Risk, $\mu_\theta = [',...
    vec_str(1:end-1),']^T$'],'Interpreter','latex');
xlabel('$N$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
legend(sub_leg,str_leg,'Interpreter','latex','Location','northeast');


figure(2); clf;
sub_leg = [];
str_leg = cell(1,numel(N_p));
for idx = 1:numel(N_p)
    hold on;
    p_a = plot(alpha_0_p,R_a(idx,:)','-','Color',colors(1+mod(idx-1,N_clr),:));
    p_e = plot(alpha_0_p,R_e(idx,:)','--','Color',colors(1+mod(idx-1,N_clr),:)); 
    
    sub_leg = [sub_leg, p_a];    
    str_leg(idx) = {['$N = ',num2str(N_p(idx)),'$']};
end
grid on; %set(gca,'YLim',[0,1]);
vec_str = num2str(mu_theta','%0.1f,'); title([title_loss,' Risk, $\mu_\theta = [',...
    vec_str(1:end-1),']^T$'],'Interpreter','latex');
xlabel('$\alpha_0$','Interpreter','latex'); 
ylabel('$\mathcal{R}(f)$','Interpreter','latex'); 
legend(sub_leg,str_leg,'Interpreter','latex','Location','southeast');





% figure(1); clf;
% mesh(alpha_0_p,N_p,R_e);
% grid on; xlabel('alpha_0'); ylabel('N'); zlabel('Risk');

% plot(N_p,R_a,'o',N_p,R_e,'*'); 
% plot(N_p,R_e,'*');
% hold on; plot(N_p,R_a,'o'); 

% figure(2); clf;
% mesh(alpha_0_p,N_p,(R_e-R_a)./R_a);
% grid on; xlabel('alpha_0'); ylabel('N'); zlabel('Risk');
% 
% figure(3); clf;
% mesh(alpha_0_p,N_p,(R_e-R_b)./R_a);
% grid on; xlabel('alpha_0'); ylabel('N'); zlabel('Risk');


% figure(1); clf;
% mesh(alpha_0_p,N_p,(R_g-R_b)./R_g);
% grid on; xlabel('alpha_0'); ylabel('N'); zlabel('Risk');
% 
% figure(2); clf;
% mesh(alpha_0_p,N_p,R_g);
% grid on; xlabel('alpha_0'); ylabel('N'); zlabel('Risk');
% 
% figure(3); clf;
% mesh(alpha_0_p,N_p,R_b);
% grid on; xlabel('alpha_0'); ylabel('N'); zlabel('Risk');





