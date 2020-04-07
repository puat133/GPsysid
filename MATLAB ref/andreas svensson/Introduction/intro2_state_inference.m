% Andreas Svensson, 2016

clear

% In this script, we consider the case of having a perfectly known
% one-dimensional model, and inferring the states x (akin to Kalman filter)

%% Generate data

% First choose the model and generate T data samples from it

T = 30;
f = @(x) -3*atan(x*5);
g = @(x) x;

Q = 3;
R = 0.1;

x_true = zeros(1,T+1);
y = zeros(1,T);
for t = 1:T
    x_true(1,t+1) = f(x_true(1,t)) + mvnrnd(0,Q);
    y(1,t) = g(x_true(1,t)) + mvnrnd(0,R);
end

%% State inference
K = 1000;
N = 20;

% Some memory allocation
x_prim = zeros(1,1,T,K);

% Run Algorithm 1
for k = 1:K
    
   % Pre-allocate
    w = zeros(T,N); x_pf = zeros(1,N,T);
    a = zeros(T,N);

    % Initialize
    if k > 1; x_pf(:,end,:) = x_prim(:,:,:,k-1); end
    w(1,:) = 1; w(1,:) = w(1,:)./sum(w(1,:));
    
    % CPF with ancestor sampling %
    
    x_pf(:,1:end-1,1) = 0;
    
    for t = 1:T
        
        % PF time propagation, resampling and ancestor sampling
        if t >= 2
            if k > 1 % Run the conditional particle filter with ancestor sampling
                a(t,1:N-1) = systematic_resampling(w(t-1,:),N-1);
                x_pf(:,1:N-1,t) = f(x_pf(:,a(t,1:N-1),t-1)) + mvnrnd(zeros(N-1,1),Q)';

                waN = w(t-1,:).*mvnpdf(f(x_pf(:,:,t-1))',x_pf(:,N,t)',Q)';
                waN = waN./sum(waN); a(t,N) = systematic_resampling(waN,1);
            else % Run a standard PF on first iteration
                a(t,:) = systematic_resampling(w(t-1,:),N);
                x_pf(:,:,t) = f(x_pf(:,a(t,:),t-1)) + mvnrnd(zeros(N,1),Q)';
            end
        end
               
        % PF weight update (work with logarithms for numerical reasons)
        log_w = -(g(x_pf(:,:,t)) - y(t)).^2/2/R; 
        w(t,:) = exp(log_w - max(log_w));
        w(t,:) = w(t,:)/sum(w(t,:));
    end

    % Sample trajectory to condition on
    
    star = systematic_resampling(w(end,:),1);
    x_prim(:,1,T,k) = x_pf(:,star,T);
    for t = T-1:-1:1
        star = a(t+1,star);
        x_prim(:,1,t,k) = x_pf(:,star,t);
    end

    if round(k/100)*100 == k; display(['Iteration k = ', num2str(k)]); end
end


%% Plot the result

figure(3), clf

% Remove burn-in from MCMC procedures
burn_in = min(floor(K/2),2000);

sh = plot(1:T,squeeze(x_prim(:,:,:,burn_in+1:K)),'k');
hold on
th = plot(1:T,x_true(1,1:T),'r');
title('State inference with Algorithm 1')
xlabel('t')
ylabel('x')
legend([sh(1) th],'Samples from p(x(1:T)|y(1:T))','True x')