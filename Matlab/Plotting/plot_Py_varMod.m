%%%% P(y,x) prior to posterior

clear;

%%% Inputs

M_y = 10;
M_x = 1;

N_vec = [1,10,100];                     % Number of training data
% N_vec = 0;

% alpha_0_vec = [.1,10];
alpha_0_vec = [10];



% P_yx = ones(M_y,M_x)/(M_y*M_x);

p_a = 0.25;
p_t = 0.75;

P_yx = zeros(M_y,1);
for idx = 1:M_y
    P_yx(idx) = nchoosek(M_y-1,idx-1)*(p_a)^(idx-1) * (1-p_a)^(M_y-idx);
end





%%% Model
theta = zeros(M_y,1);
for idx = 1:M_y
    theta(idx) = nchoosek(M_y-1,idx-1)*(p_t)^(idx-1) * (1-p_t)^(M_y-idx);
end

% temp = zeros(M_y,M_x);
% for m_y = 1:M_y
%     for m_x = 1:M_x
%         temp(m_y,m_x,:) = gamrnd(alpha(m_y,m_x),1);
%     end
% end
% theta = temp ./ repmat(sum(sum(temp,2),1),[M_y,M_x]); 




%%% Training Data  
P_post = zeros(M_y,M_x,numel(N_vec),numel(alpha_0_vec));

bias_post = zeros(M_y,M_x,numel(N_vec),numel(alpha_0_vec));
var_post = zeros(M_y,M_x,numel(N_vec),numel(alpha_0_vec));
err_post = zeros(M_y,M_x,numel(N_vec),numel(alpha_0_vec));

var_pos = zeros(M_y,M_x,numel(N_vec),numel(alpha_0_vec));
var_neg = zeros(M_y,M_x,numel(N_vec),numel(alpha_0_vec));


for idx_n = 1:numel(N_vec)
    for idx_a = 1:numel(alpha_0_vec)
        N = N_vec(idx_n);
        alpha_0 = alpha_0_vec(idx_a);
        alpha = alpha_0*P_yx;         % prior PDF parameters
        
        P_post(:,:,idx_n,idx_a) = (alpha + N*theta) / (alpha_0 + N);

        bias_post(:,:,idx_n,idx_a) = P_post(:,:,idx_n,idx_a) - theta;
        var_post(:,:,idx_n,idx_a) = N*(alpha_0 + N)^(-2) * theta .* (1-theta);
        err_post(:,:,idx_n,idx_a) = bias_post(:,:,idx_n,idx_a).^2 + var_post(:,:,idx_n,idx_a);
        
        
        temp = zeros(M_y,M_x);
        for ii_t = 1:numel(theta)
            i_pos = (ceil(N*theta(ii_t)):N);

            for ii = i_pos
                temp(ii_t,1) = temp(ii_t,1) + nchoosek(N,ii) * theta(ii_t).^ii ...
                    .* (1-theta(ii_t)).^(N-ii) .* (ii-N*theta(ii_t)).^2;
            end
            var_pos(ii_t,:,idx_n,idx_a) = (alpha_0 + N)^(-2) * temp(ii_t,1);

            var_neg(ii_t,:,idx_n,idx_a) = var_post(ii_t,:,idx_n,idx_a) - var_pos(ii_t,:,idx_n,idx_a);
        end
        
        
    end
end


% N_bar = zeros(M_y,M_x);
% 
% P_post = [];
% if N_vec(1) == 0
%     P_post = cat(3,P_post,(alpha+N_bar) / sum(alpha(:) + N_bar(:)));
% end
% 
% for n = 1:N_vec(end)
%     temp = find(rand <= cumsum(theta(:)));
%     idx_d = temp(1);
%     [idx_d_y,idx_d_x] = ind2sub([M_y,M_x],idx_d); 
% 
%     N_bar(idx_d_y,idx_d_x) = N_bar(idx_d_y,idx_d_x) + 1;
%     
%     if ismember(n,N_vec)
%         P_post = cat(3,P_post,(alpha+N_bar) / sum(alpha(:) + N_bar(:)));
%     end
% end




%%% Plot

Y_set = cell(M_y,1);
for idx = 1:M_y
    Y_set{idx} = ['$\mathcal{Y}_{',num2str(idx),'}$'];
%     Y_set{idx} = ['$(\mathcal{Y}_{',num2str(idx),'},\mathcal{X}_1)$'];
end


figure(1); clf; 
for idx_n = 1:numel(N_vec)
    for idx_a = 1:numel(alpha_0_vec)
        subplot(numel(N_vec),numel(alpha_0_vec),idx_a+(idx_n-1)*numel(alpha_0_vec));
%         subplot(numel(alpha_0_vec),numel(N_vec),idx_n+(idx_a-1)*numel(N_vec));
        hold on;
        plot(1:M_y,theta,'*k')
%         errorbar(1:M_y,P_post(:,:,idx_n,idx_a),sqrt(var_post(:,:,idx_n,idx_a)),'o');
        errorbar(1:M_y,P_post(:,:,idx_n,idx_a),sqrt(var_neg(:,:,idx_n,idx_a)),sqrt(var_pos(:,:,idx_n,idx_a)),'o');
        grid on; axis([1,M_y,0,0.4]);
%         title(['$\alpha_0 = ',num2str(alpha_0_vec(idx_a)),'$, $N = ',num2str(N_vec(idx_n)),'$'],'Interpreter','latex'); 
%         title(['$\alpha_0 = ',num2str(alpha_0_vec(idx_a)),'$, $N = ',num2str(N_vec(idx_n)),'$; $\mathrm{Error} = ',...
%             num2str(sqrt(sum(sum(err_post(:,:,idx_n,idx_a)))),'%0.3f'),'$'],'Interpreter','latex'); 
%         title(['$N = ',num2str(N_vec(idx_n)),'$; $\mathrm{Error} = ',...
%             num2str(sqrt(sum(sum(err_post(:,:,idx_n,idx_a)))),'%0.3f'),'$'],'Interpreter','latex'); 
        title(['$\alpha''(x) = ',num2str(alpha_0_vec(idx_a)),'$, $\mathrm{n}''(x) = ',num2str(N_vec(idx_n)),'$; $\mathrm{Error} = ',...
            num2str(sqrt(sum(sum(err_post(:,:,idx_n,idx_a)))),'%0.3f'),'$'],'Interpreter','latex'); 
        xticks(1:M_y); set(gca,'TickLabelInterpreter','latex'); xticklabels(Y_set'); xtickangle(0);
        xlabel('$y$','Interpreter','latex');
%         legend({'$\theta(y,x)$','P$(y,x|\mathrm{D})$'},'Interpreter','latex',...
%             'Location','Northwest');
        legend({'P$(y|\mathrm{x},\theta)$','P$(y|\mathrm{x},\mathrm{D})$'},'Interpreter','latex',...
            'Location','Northwest');
    end
end


% figure(1); clf; 
% for idx = 1:numel(N_vec)
%     subplot(numel(N_vec),1,idx);
%     hold on;
%     plot(1:M_y,theta,'*k',1:M_y,P_post(:,:,idx),'o');
%     grid on; axis([1,M_y,0,0.5]);
%     title(['$N = ',num2str(N_vec(idx)),'$'],'Interpreter','latex'); 
%     xticks(1:M_y); set(gca,'TickLabelInterpreter','latex'); xticklabels(Y_set'); 
% %     ylabel('P$(\mathrm{y},\mathrm{x} | \mathrm{D})$','Interpreter','latex');
%     legend({'$\theta(y,x)$','P$(y,x|\mathrm{D})$'},'Interpreter','latex');
% end

