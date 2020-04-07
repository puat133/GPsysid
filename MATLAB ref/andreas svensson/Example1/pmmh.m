function [ theta ] = pmmh( K, N, n_th, u, y, f_g, g, nx, prior_sample, prior_pdf, Q, R)
%PMH 
% Andreas Svensson, 2016


theta = zeros(n_th,K);
log_W = -Inf;

while log_W==-Inf %Find an initial sample without numerical problems
    theta(:,1) = prior_sample(1);
    log_W = pf( N, @(x,u)f_g(x,u,theta(:,1)'), g, u, y, Q, R, nx);
end

k = 1;
acc = 0;
while acc < K
    theta_prop = theta(:,k) + (1-2*rand(size(theta(:,k))));
    log_W_prop = pf( N, @(x,u)f_g(x,u,theta_prop'), g, u, y, Q, R, nx);
    dm = rand;
    mh_ratio = exp(log_W_prop-log_W)*prior_pdf(theta_prop')/prior_pdf(theta(:,k)');
    if isnan(mh_ratio)
        alpha = 0;
    else
        alpha = min(1,mh_ratio);
    end
    if dm < alpha
        theta(:,k+1) = theta_prop;
        log_W = log_W_prop;
        acc = acc+1;
    else
        theta(:,k+1) = theta(:,k);
    end
    if round(k/10)*10 == k
        display(['PMH Sampling ', num2str(k), ': ',num2str(acc),' accepted']);
    end
    k = k+1;
    if k == length(theta)
        theta = [theta,zeros(n_th,K)]; %#ok
    end
end

theta = theta(:,1:k);

end

