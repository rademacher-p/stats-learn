function Risk = risk_min_dir_01(Y,X,N,alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Min. Bayes Risk, Dirichlet Prior, 0-1 Loss
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M_y = numel(Y);
M_x = numel(X);

    
if sum(alpha(:) == 1) == numel(alpha)       %%% Uniform Dirichlet prior

    temp = 0;
    for m = 1:M_y
        temp2 = 0;
        for n = 0:floor(N/m)
            temp2 = temp2 + prod(1 - m*n./(N+(1:M_y*M_x-1)));
        end
        temp = temp + nchoosek(M_y,m) * (-1)^(m-1) * temp2;
    end 
    Risk = 1 - temp / (M_y + N/M_x);
    
else
    
    alpha_0 = sum(alpha(:));

    temp = 0;
    for ii_x = 1:M_x
        
    
        alpha_m = sum(alpha(:,ii_x));
        alpha_s = alpha(:,ii_x);
        
        if M_x == 1 %%%%%
            beta_m = 1;
        else
            beta_m = beta(alpha_m,alpha_0-alpha_m);
        end
        
        beta_s = beta_multi(alpha_s);
        
        
        temp_m = 0;
        for n_m = 0:N

            N_bar = N_bar_set_gen([M_y,1],n_m); 
                                          
            temp_c = 0;
            for ii_c = 1:size(N_bar,3)
                
                n_bar_s = N_bar(:,1,ii_c);
                
                DM_pdf_c = coef_multi(n_bar_s) * beta_multi(alpha_s + n_bar_s) / beta_s;            
                temp_c = temp_c + DM_pdf_c * max(alpha_s + n_bar_s);
            end
            
            if M_x == 1 %%%%%
                DM_pdf_m = 1*(n_m == N);
            else
                DM_pdf_m = nchoosek(N,n_m) * beta(alpha_m+n_m,alpha_0-alpha_m+N-n_m) / beta_m;
            end
            
            temp_m = temp_m + DM_pdf_m * temp_c;
        end

        temp = temp + temp_m;       
    end
    
    Risk = 1 - temp / (alpha_0 + N);
    
end

