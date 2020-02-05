function N_bar = N_bar_set_gen(M_YX,N)

M = prod(M_YX);

if M == 1
    N_bar = N;
else
    
    N_bar = (0:N);

    for m = 2:M-1
        temp = N_bar;
        N_bar = [];
        for idx = 1:size(temp,2)
            N_bar = [N_bar, [temp(:,idx)*ones(1,N-sum(temp(:,idx))+1) ; 0:(N-sum(temp(:,idx)))] ];
        end
    end

    N_bar = [N_bar ; N-sum(N_bar,1)];

end

N_bar = reshape(N_bar,[M_YX,size(N_bar,2)]);