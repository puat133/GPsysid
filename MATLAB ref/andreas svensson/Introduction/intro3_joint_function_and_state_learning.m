% Andreas Svensson, 2016

clear
% In this script, we combine the two previous scripts for learning the
% model (i.e., the function f and Q) and the states JOINTLY
% (we assume the observation function g is known, as well as R)

%% Generate data

% First choose the true model (to be learned) and generate T data samples
% from it

T = 30;
f_true = @(x) -2*atan(x*5);
g = @(x) x;

Q_true = 0.1;
R = 0.1;

x_true = zeros(1,T+1);
y = zeros(1,T);
for t = 1:T
    x_true(1,t+1) = f_true(x_true(1,t)) + mvnrnd(0,Q_true);
    y(1,t) = g(x_true(1,t)) + mvnrnd(0,R);
end

xv = linspace(1.8*min(x_true),1.8*max(x_true),200);

clf
plot(x_true(1:T-1),x_true(2:T),'.')
hold on
plot(xv,f_true(xv))

%% Set up priors
% ... as in script 1

n_basis = 20; % Number of basis functions
L = 20;       % The interval

jv = (1:n_basis)';
phi = @(x) L^(-1/2)*sin(pi*bsxfun(@times,jv,(x+L))./(2*L)); % The choice of basis functions
lambda = @(j) (pi*j/(2*L)).^2; % Their eigenvalues (for the GP-inspired priors)

% Some covariance functions
S_SE = @(w,ell,sf) sf*sqrt(2*pi.*ell.^2).*exp(-(w.^2./2).*ell.^2); % Spectral density for the exponentiated quadratic
S_M  = @(w,ell,ny,sf) sf*(2*pi^(1/2)*gSA(ny+1/2)*(2*ny).^ny)/(gSA(ny)*ell.^(2*ny))*(2*ny/ell.^2 + w.^2).^(-ny-1/2); % Spectral density for the Matérn covariance function

% Define the matrix V (here, we choose covariance function and its hyperparameters)
V = @(n) diag(repmat(S_SE(sqrt(lambda(jv')),3,10),[1 n+1]));

% Priors for Q
lQ = 1; LambdaQ = 5;

% Prior for the occurences of discontinuity points (geometrically
% distributed, with parameter p. p = 1: no discontinuity points)
p = 0.7;



%% Sample from prior
% A good practice is to first sample from the prior, before running the
% learning algorithm, as a sanity-check.

figure(4), clf
for i = 1:5

    np = geornd(p);
    ptsp = sort([-L*100, L*(1-2*rand([1 np])), L*100]);
    modelp = gibbs_param(0, 0, 0, V(np), LambdaQ,lQ,0);
    
    fp = @(x) modelp.A*(bsxfun(@ge,repmat(x(1,:),[n_basis*(np+1) 1]),kron(ptsp(1:end-1)',ones(n_basis,1))).*bsxfun(@lt,repmat(x(1,:),[n_basis*(np+1) 1]),kron(ptsp(2:end)',ones(n_basis,1))).*repmat(phi(x),np+1,1));
    
    ph = plot(xv,fp(xv),'k');
    hold on
end
th = plot(xv,f_true(xv),'r');
xlabel('z')
ylabel('f(z)')
legend([ph, th],'Samples from prior','True function')


%% Learning alternative 1
K = 4000; N = 20;

% Some memory allocation
model1 = cell(K,1);
x_prim1 = zeros(1,1,T,K);

% Initialization
model1{1}.A = zeros(1,n_basis);
model1{1}.Q = 1;

% Run the Bayesian learning, Algorithm 2, WITHOUT discontinuity points
for k = 1:K
   
    %%%%%%%%%%
    % Step 3 %
    %%%%%%%%%%
    
    A = model1{k}.A; Q = model1{k}.Q;
    f = @(x) A*phi(x);
    
    % Pre-allocate
    w = zeros(T,N); x_pf = zeros(1,N,T);
    a = zeros(T,N);

    % Initialize
    if k > 1; x_pf(:,end,:) = x_prim1(:,:,:,k-1); end
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
    x_prim1(:,1,T,k) = x_pf(:,star,T);
    for t = T-1:-1:1
        star = a(t+1,star);
        x_prim1(:,1,t,k) = x_pf(:,star,t);
    end
    
    % As we have no discontinuity points, we ignore step 4
    
    %%%%%%%%%%%%%%
    % Step 5 & 6 %
    %%%%%%%%%%%%%%
 
    % Compute statistics
    zeta = squeeze(x_prim1(:,:,2:T,k))';
    z = squeeze(x_prim1(:,:,1:T-1,k))';
    Phi = zeta*zeta'; Psi = zeta*phi(z)'; Sig = phi(z)*phi(z)';
    
    % Sample new A and Q for the new model
    model1{k+1} = gibbs_param( Phi, Psi, Sig, V(0), LambdaQ, lQ, T-1);

    if round(k/100)*100 == k; display(['Iteration k = ', num2str(k)]); end
end


%% Learning alternative 2

% Some memory allocation
model2 = cell(K,1);
x_prim2 = zeros(1,1,T,K);

% Initialization
model2{1}.A = zeros(1,n_basis);
model2{1}.Q = 1;
model2{1}.n = 0;
model2{1}.pts = [-L,L];

% Run the Bayesian learning, Algorithm 2, WITH discontinuity points
for k = 1:K
    
    %%%%%%%%%%
    % Step 3 %
    %%%%%%%%%%
    
    A = model1{k}.A; Q = model1{k}.Q;
    f = @(x) A*phi(x);
    
    % Pre-allocate
    w = zeros(T,N); x_pf = zeros(1,N,T);
    a = zeros(T,N);

    % Initialize
    if k > 1; x_pf(:,end,:) = x_prim2(:,:,:,k-1); end
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
    x_prim2(:,1,T,k) = x_pf(:,star,T);
    for t = T-1:-1:1
        star = a(t+1,star);
        x_prim2(:,1,t,k) = x_pf(:,star,t);
    end
    
    %%%%%%%%%%
    % Step 4 %
    %%%%%%%%%%
    
    zeta = squeeze(x_prim1(:,:,2:T,k))';
    z = squeeze(x_prim1(:,:,1:T-1,k))';
    
    % Propose a new jump model
    np = geornd(p);
    ptsp = sort([-L*100, L*(1-2*rand([1 np])), L*100]);
    
    % Compute the statistics and marginal likelihood for the proposed model (prop)
    zp = bsxfun(@ge,repmat(z,[n_basis*(np+1) 1]),kron(ptsp(1:end-1)',ones(n_basis,1))).*bsxfun(@lt,repmat(z,[n_basis*(np+1) 1]),kron(ptsp(2:end)',ones(n_basis,1))).*repmat(phi(z),np+1,1);
    prop.Phi = zeta*zeta'; prop.Psi = zeta*zp'; prop.Sig = zp*zp';
    prop.V = V(np);
    prop.marginal_likelihood = compute_marginal_likelihood(prop.Phi,prop.Psi,prop.Sig,prop.V,LambdaQ,lQ,T-1);
    prop.n = np; prop.pts = ptsp;
    
    if k > 1
        % Alternatively staying with the current model (curr)
        np = model2{k}.n; ptsp = model2{k}.pts;

        % Compute its statistics and marginal likelihood
        zp = bsxfun(@ge,repmat(z,[n_basis*(np+1) 1]),kron(ptsp(1:end-1)',ones(n_basis,1))).*bsxfun(@lt,repmat(z,[n_basis*(np+1) 1]),kron(ptsp(2:end)',ones(n_basis,1))).*repmat(phi(z),np+1,1);
        curr.Phi = zeta*zeta'; curr.Psi = zeta*zp'; curr.Sig = zp*zp';
        curr.M = zeros(1,(np+1)*n_basis); curr.V = V(np);
        curr.marginal_likelihood = compute_marginal_likelihood(curr.Phi,curr.Psi,curr.Sig,curr.V,LambdaQ,lQ,T-1);
        curr.n = np; curr.pts = ptsp;
    end
    
    dv = rand; % Perform Metropolis-Hastings step
    if (k == 1) || (dv < min(exp(prop.marginal_likelihood - curr.marginal_likelihood),1))
        nmodel = prop;
    else
        nmodel = curr;
    end

    
    %%%%%%%%%%%%%%
    % Step 5 & 6 %
    %%%%%%%%%%%%%%
    
    % Sample new A and Q for the new model
    model2{k+1} = gibbs_param( nmodel.Phi, nmodel.Psi, nmodel.Sig, nmodel.V, LambdaQ,lQ,T-1);
    model2{k+1}.n = nmodel.n; model2{k+1}.pts = nmodel.pts;

    if round(k/100)*100 == k; display(['Iteration k = ', num2str(k)]); end
end


%% Learning alternative 3

% Run the Maximum likelihood learning, Algorithm 3, WITH regularization

% Some memory allocation
model4 = cell(K,1);
x_prim4 = zeros(1,1,T,K);

% Initialization
model4{1}.A = zeros(1,n_basis);
model4{1}.Q = 1;

Phi_k = 0; Psi_k = zeros(1,n_basis); Sig_k = zeros(n_basis,n_basis);
% Define a gamam sequence for stochastic approximation
gSA = zeros(1,K);gSA(1:20) = 1; gSA(21:50) = 0.9;
gSA(51:end) = 0.9*(((0:K-51)+1)/1).^(-0.7);

% Run the Bayesian learning, Algorithm 2, WITHOUT discontinuity points
for k = 1:K
   
    %%%%%%%%%%
    % Step 3 %
    %%%%%%%%%%
    
    A = model4{k}.A; Q = model4{k}.Q;
    f = @(x) A*phi(x);
    
    % Pre-allocate
    w = zeros(T,N); x_pf = zeros(1,N,T);
    a = zeros(T,N);

    % Initialize
    if k > 1; x_pf(:,end,:) = x_prim4(:,:,:,k-1); end
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
    x_prim4(:,1,T,k) = x_pf(:,star,T);
    for t = T-1:-1:1
        star = a(t+1,star);
        x_prim4(:,1,t,k) = x_pf(:,star,t);
    end
    
    %%%%%%%%%%
    % Step 4 %
    %%%%%%%%%%
    
    % Compute statistics
    zeta = squeeze(x_prim4(:,:,2:T,k))';
    z = squeeze(x_prim4(:,:,1:T-1,k))';
    Phi = zeta*zeta'; Psi = zeta*phi(z)'; Sig = phi(z)*phi(z)';
    
    Phi_k = (1-gSA(k))*Phi_k + gSA(k)*Phi;
    Psi_k = (1-gSA(k))*Psi_k + gSA(k)*Psi;
    Sig_k = (1-gSA(k))*Sig_k + gSA(k)*Sig;
    
    %%%%%%%%%%
    % Step 5 %
    %%%%%%%%%%
 
    % Compute new A and Q for the new model
    model4{k+1} = saem_param( Phi_k, Psi_k, Sig_k, V(0), LambdaQ, lQ, T-1, n_basis);

    if round(k/100)*100 == k; display(['Iteration k = ', num2str(k)]); end
end


%% Plot the result

figure(5), clf
axv = 1.8*[min(x_true) max(x_true) min(x_true) max(x_true)];

% Remove burn-in from MCMC procedures
burn_in = min(floor(K/2),2000);


% Plot the MCMC samples from learning 1
subplot(221)
funcv1 = zeros(K-burn_in+1,length(xv));
for k = burn_in:K
    func1 = @(x) model1{k}.A*phi(x);
    funcv1(k-burn_in+1,:) = func1(xv);
end
sh = plot(xv,funcv1,'k');
hold on
th = plot(xv,f_true(xv),'--g','linewidth',2);
dh = plot(x_true(1:T-1),x_true(2:T),'r.','markersize',30);
axis(axv)
title('Fully Bayesian learning without discontinuity points')
xlabel('z')
ylabel('f(z)')
legend([sh(1) dh th],'Samples from posterior','Underlying state samples (not available for learning)','True function')
drawnow


% Plot the MCMC samples from learning 2
subplot(222)
funcv2 = zeros(K-burn_in+1,length(xv));
for k = burn_in:K
    n = model2{k}.n;
    pts = model2{k}.pts;
    func2 = @(x) model2{k}.A*(bsxfun(@ge,repmat(x,[n_basis*(n+1) 1]),kron(pts(1:end-1)',ones(n_basis,1))).*bsxfun(@lt,repmat(x,[n_basis*(n+1) 1]),kron(pts(2:end)',ones(n_basis,1))).*repmat(phi(x),n+1,1));
    funcv2(k-burn_in+1,:) = func2(xv);
end
sh = plot(xv,funcv2,'k');
hold on
th = plot(xv,f_true(xv),'--g','linewidth',2);
dh = plot(x_true(1:T-1),x_true(2:T),'r.','markersize',30);
axis(axv)
title('Fully Bayesian learning with discontinuity points')
xlabel('z')
ylabel('f(z)')
legend([sh(1) dh th],'Samples from posterior','Underlying state samples (not available for learning)','True function')
drawnow


% Plot the ML estimate from learning 3
subplot(223)
func4 = @(x) model4{K}.A*phi(x);
eh = plot(xv,func4(xv),'k');
hold on
th = plot(xv,f_true(xv),'--g','linewidth',2);
dh = plot(z,zeta,'r.','markersize',30);
axis(axv)
title('Maximum Likelihood learning with regularization (no discontinuity points)')
xlabel('z')
ylabel('f(z)')
legend([eh dh th],'Regularized maximum likelihood estimate','Data points','True function')