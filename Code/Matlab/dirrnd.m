function out = dirrnd(alpha,L)

M_y = size(alpha,1);
M_x = size(alpha,2);

temp = zeros(M_y,M_x,L);
for m_y = 1:M_y
    for m_x = 1:M_x
        temp(m_y,m_x,:) = gamrnd(alpha(m_y,m_x),1,[1,1,L]);
    end
end
out = temp ./ repmat(sum(sum(temp,2),1),[M_y,M_x]); 
