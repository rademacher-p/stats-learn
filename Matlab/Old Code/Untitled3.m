%%%%%

clear;

Y = num2cell((1:2)'/2);             % output set
X = num2cell((1:3)');             % input set


% M = 3;
% N_t = 100;
% Theta = N_bar_set(M,N_t)/N_t;
% L_set = size(Theta,2);




N_t = 100;
% P_y_x = N_bar_set(numel(Y)*numel(X),N_t)/N_t;
P_x = N_bar_set(numel(X),N_t)/N_t;

alpha_0 = numel(Y)*numel(X);
% alpha_0 = .1;

N = 10;


% Risk_a = risk_opt_SE(Y,alpha,N);

sc = zeros(size(P_x,2),1);
for i_t = 1:size(P_x,2)
    sc(i_t) = P_x(:,i_t)' * ((P_x(:,i_t)+1/(alpha_0+N))./(P_x(:,i_t)+1/alpha_0));
end



figure(1); clf;
scatter3(P_x(1,:),P_x(2,:),P_x(3,:),100,sc,'.');
xlabel('P$(x_1)$','Interpreter','latex'); 
ylabel('P$(x_2)$','Interpreter','latex'); 
zlabel('P$(x_3)$','Interpreter','latex'); 
cbar = colorbar; cbar.Label.Interpreter = 'latex'; cbar.Label.String = 'scale';
title(['$N = ',num2str(N),'$, $\alpha_0 = ',num2str(alpha_0),'$'],'Interpreter','latex');
grid on; axis equal; view(135,45); 




%%%%%

P_x = ones(size(X))/numel(X);
% P_x = [1; zeros(numel(X)-1,1)];

% alpha_0_p = [.1, 1, 10]';
alpha_0_p = (0.5:0.5:20)';

% N_p = (0:25)';
N_p = [0, 1, 10]';


sc = zeros(numel(N_p),numel(alpha_0_p));
for i_n = 1:numel(N_p)
    N = N_p(i_n);
    for i_a = 1:numel(alpha_0_p)
        alpha_0 = alpha_0_p(i_a);
        
%         sc(i_n,i_a) = P_x' * ((P_x+1/(alpha_0+N))./(P_x+1/alpha_0));
        sc(i_n,i_a) = ((1/3+1/(alpha_0+N))./(1/3+1/alpha_0));
    end
end


figure(2); clf;
plot(N_p,sc,'*');
xlabel('$N$','Interpreter','latex'); 
ylabel('scale','Interpreter','latex'); 
grid on; legend;

figure(3); clf;
plot(alpha_0_p,sc','*');
xlabel('$N$','Interpreter','latex'); 
ylabel('scale','Interpreter','latex'); 
grid on; legend;

