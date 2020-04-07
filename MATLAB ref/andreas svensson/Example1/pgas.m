function [ theta ] = pgas( K, N, n_th, u, y, phiv, g, nx, prior_sample, prior_V, Q, R)
%PMH 
% Andreas Svensson, 2016

T = length(y);
theta = zeros(n_th,K+1);
theta(:,1) = prior_sample(1);

x_prim = zeros(nx,1,T);

Q_chol = chol(Q);

for k = 1:K
    
    drawnow

    f_i = @(x,u) theta(:,k)'*phiv(x,u(:,ones(1,size(x,2))));
    
    
    % Pre-allocate
    w = zeros(T,N); x_pf = zeros(nx,N,T);
    a = zeros(T,N);

    % Initialize
    if k > 1; x_pf(:,end,:) = x_prim; end
    w(1,:) = 1; w(1,:) = w(1,:)./sum(w(1,:));
    
    % CPF with ancestor sampling %
    
    x_pf(:,1:end-1,1) = 0;
    
    for t = 1:T
        
        % PF time propagation, resampling and ancestor sampling
        if t >= 2
            if k > 1
                a(t,1:N-1) = systematic_resampling(w(t-1,:),N-1);
                x_pf(:,1:N-1,t) = f_i(x_pf(:,a(t,1:N-1),t-1),u(t-1)) + Q_chol*randn(nx,N-1);

                waN = w(t-1,:).*mvnpdf(f_i(x_pf(:,:,t-1),u(t-1))',x_pf(:,N,t)',Q)';
                waN = waN./sum(waN); a(t,N) = systematic_resampling(waN,1);
            else % Run a standard PF on first iteration
                a(t,:) = systematic_resampling(w(t-1,:),N);
                x_pf(:,:,t) = f_i(x_pf(:,a(t,:),t-1),u(t-1)) + Q_chol*randn(nx,N);
            end
        end
               
        % PF weight update
        log_w = -(g(x_pf(:,:,t),u(t)) - y(t)).^2/2/R; 
        w(t,:) = exp(log_w - max(log_w));
        w(t,:) = w(t,:)/sum(w(t,:));
    end
    
    drawnow

    % Sample trajectory to condition on
    
    star = systematic_resampling(w(end,:),1);
    x_prim(:,1,T) = x_pf(:,star,T);
    for t = T-1:-1:1
        star = a(t+1,star);
        x_prim(:,1,t) = x_pf(:,star,t);
    end
    
    if round(k/10)*10 == k
        display(['PGAS sampling ',num2str(k)])
    end
    
	% Compute statistics
    ze = x_prim(1,1,2:T);
    
    z = permute(phiv(squeeze(x_prim(1,1,1:T-1))',u(1:T-1)),[1 3 2]);

%     Phi1 = sum(ze.*permute(ze,[2 1 3]),3);
    Psi1 = sum(ze(:,ones(n_th,1),:).*permute(z,[2 1 3]),3);
    Sigma1 = sum(z(:,ones(n_th,1),:).*permute(z(:,ones(n_th,1),:),[2 1 3]),3);

    
    % Sample new parameters

    Sigma1_bar = Sigma1 + inv(prior_V);
    Psi1_bar = Psi1;
    
    Gamma1_star = Psi1_bar/Sigma1_bar;
    Sigma1_bar_inv = inv(Sigma1_bar);
    
    Sigma1_bar_inv_fix = 0.5*(Sigma1_bar_inv + Sigma1_bar_inv');
    
    X = randn(1,n_th);
    theta(:,k+1) = Gamma1_star + Q_chol*X*chol(Sigma1_bar_inv_fix);
    
    
end


end

