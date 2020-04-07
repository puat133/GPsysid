function [ log_W, x_pf_t, log_w_t ] = pf( N, f, g, u, y, Q, R, nx)
%PF 
% Andreas Svensson, 2016

T = length(y);
log_w = zeros(T,N);
x_pf = zeros(nx,N,T);

Q_chol = chol(Q);

for t = 1:T
    if t >= 2
        a = systematic_resampling(wn,N);
        x_pf(:,:,t) = f(x_pf(:,a,t-1),u(t-1,:)) + Q_chol*randn(nx,N);
    end
    log_w(t,:) = mvnpdf_log(g(x_pf(:,:,t),u(t,:))',y(t,:),R)'; 
    wn = exp(log_w(t,:) - max(log_w(t,:)));
    wn = wn/sum(wn);
end

x_pf_t = x_pf(:,:,T);
log_w_t = log_w(T,:);

log_W = sum(log(1/N*sum(exp(log_w),2)));

end

