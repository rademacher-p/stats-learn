%%%

clear;

M = 5*ones(1,2);


t_1 = [0,1,0;
       1,1,1;
       0,1,0];
   
t_2 = [0,1,1;
       0,1,0;
       1,1,0];
   
   

x_1 = zeros(M);
s_t = size(t_1);
d = [randi([1,M(1)-s_t(1)+1],[1,1]), randi([1,M(2)-s_t(2)+1],[1,1])];
x_1(d(1):d(1)+s_t(1)-1,d(2):d(2)+s_t(2)-1) = t_1;

x_2 = zeros(M);
s_t = size(t_2);
d = [randi([1,M(1)-s_t(1)+1],[1,1]), randi([1,M(2)-s_t(2)+1],[1,1])];
x_2(d(1):d(1)+s_t(1)-1,d(2):d(2)+s_t(2)-1) = t_2;


figure(1); clf;

subplot(2,1,1);
imagesc(x_1);
axis square;
colormap('gray');
title('$\mathrm{y} = \mathcal{Y}_1$','Interpreter','latex');

subplot(2,1,2);
imagesc(x_2);
axis square;
colormap('gray');
title('$\mathrm{y} = \mathcal{Y}_2$','Interpreter','latex');

