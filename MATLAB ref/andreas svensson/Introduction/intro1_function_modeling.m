% Andreas Svensson, 2016

clear
% We start in this script by modeling a one-dimensional static function f,
% when we have noisy samples from it (i.e., there is no dynamics involved)

%% Generate data

% First choose the true function (to be learned later) and
% generate T data samples from it: f(z) + noise
T = 30;
f_true = @(x) -2*atan(x*5);
z = linspace(-7,7,T) + randn(1,T);
zeta = f_true(z) + 0.1*randn(1,T);

zv = linspace(1.8*min(z),1.8*max(z),200);


%% Set up priors
% Set up the basis functions and its priors

n_basis = 20; % Number of basis functions
L = 20;       % The interval

jv = (1:n_basis)';
phi = @(x) L^(-1/2)*sin(pi*bsxfun(@times,jv,(x+L))./(2*L)); % The choice of basis functions
lambda = @(j) (pi*j/(2*L)).^2; % Their eigenvalues (for the GP-inspired priors)

% Some covariance functions
S_SE = @(w,ell,sf) sf*sqrt(2*pi.*ell.^2).*exp(-(w.^2./2).*ell.^2); % Spectral density for the exponentiated quadratic
S_M  = @(w,ell,ny,sf) sf*(2*pi^(1/2)*gamma(ny+1/2)*(2*ny).^ny)/(gamma(ny)*ell.^(2*ny))*(2*ny/ell.^2 + w.^2).^(-ny-1/2); % Spectral density for the Matérn covariance function

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

figure(1), clf
for i = 1:5

    np = geornd(p);
    ptsp = sort([-L*100, L*(1-2*rand([1 np])), L*100]);
    modelp = gibbs_param(0, 0, 0, V(np), LambdaQ,lQ,0);
    
    fp = @(x) modelp.A*(bsxfun(@ge,repmat(x(1,:),[n_basis*(np+1) 1]),kron(ptsp(1:end-1)',ones(n_basis,1))).*bsxfun(@lt,repmat(x(1,:),[n_basis*(np+1) 1]),kron(ptsp(2:end)',ones(n_basis,1))).*repmat(phi(x),np+1,1));
    
    ph = plot(zv,fp(zv),'k');
    hold on
end
th = plot(zv,f_true(zv),'r');
xlabel('z')
ylabel('f(z)')
legend([ph, th],'Samples from prior','True function')

% Now, it is time to reflect whether the priors correspond to your prior
% beliefs about f. This is the moment to adjust the priors.
% Try adjusting them by changing the definition of p and V above.



%% Learning alternative 1
K = 4000;

% Some memory allocation
model1 = cell(K,1);

% Run the Bayesian learning, Algorithm 2, WITHOUT discontinuity points
for k = 1:K
    
    % Since we consider the static case, we ignore step 3 (Algorithm 1)
    
    % As we have no discontinuity points, we ignore step 4
    
    %%%%%%%%%%%%%%
    % Step 5 & 6 %
    %%%%%%%%%%%%%%
 
    % Compute statistics
    Phi = zeta*zeta'; Psi = zeta*phi(z)'; Sig = phi(z)*phi(z)';
    
    % Sample new A and Q for the new model
    model1{k+1} = gibbs_param( Phi, Psi, Sig, V(0), LambdaQ,lQ,T);

    if round(k/100)*100 == k; display(['Iteration k = ', num2str(k)]); end
end


%% Learning alternative 2

% Some memory allocation
model2 = cell(K,1);

% Run the Bayesian learning, Algorithm 2, WITH discontinuity points
for k = 1:K
    
    % Since we consider the static case, we ignore step 3 (Algorithm 1)
    
    %%%%%%%%%%
    % Step 4 %
    %%%%%%%%%%
    
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
    model2{k+1} = gibbs_param( nmodel.Phi, nmodel.Psi, nmodel.Sig, nmodel.V, LambdaQ, lQ, T);
    model2{k+1}.n = nmodel.n; model2{k+1}.pts = nmodel.pts;

    if round(k/100)*100 == k; display(['Iteration k = ', num2str(k)]); end
end
%% Learning alternative 4

% Run the Maximum likelihood learning, Algorithm 3, WITH regularization

% Compute statistics
Phi = zeta*zeta'; Psi = zeta*phi(z)'; Sig = phi(z)*phi(z)';
model3 = saem_param( Phi, Psi, Sig, V(0), LambdaQ, lQ, T, n_basis);

%% Learning alternative 3

% Run the Maximum likelihood learning, Algorithm 3, WITHOUT regularization

% Compute statistics
Phi = zeta*zeta'; Psi = zeta*phi(z)'; Sig = phi(z)*phi(z)';
model4 = saem_param_nonreg( Phi, Psi, Sig, T);

%% Plot the result

figure(2), clf
axv = 1.8*[min(z) max(z) min(zeta) max(zeta)];

% Remove burn-in from MCMC procedures
burn_in = min(floor(K/2),2000);


% Plot the MCMC samples from learning 1
subplot(221)
funcv1 = zeros(K-burn_in+1,length(zv));
for k = burn_in:K
    func1 = @(x) model1{k}.A*phi(x);
    funcv1(k-burn_in+1,:) = func1(zv);
end
sh = plot(zv,funcv1,'k');
hold on
th = plot(zv,f_true(zv),'--g','linewidth',2);
dh = plot(z,zeta,'r.','markersize',30);
axis(axv)
title('Fully Bayesian learning without discontinuity points')
xlabel('z')
ylabel('f(z)')
legend([sh(1) dh th],'Samples from posterior','Data points','True function')
drawnow


% Plot the MCMC samples from learning 2
subplot(222)
funcv2 = zeros(K-burn_in+1,length(zv));
for k = burn_in:K
    n = model2{k}.n;
    pts = model2{k}.pts;
    func2 = @(x) model2{k}.A*(bsxfun(@ge,repmat(x,[n_basis*(n+1) 1]),kron(pts(1:end-1)',ones(n_basis,1))).*bsxfun(@lt,repmat(x,[n_basis*(n+1) 1]),kron(pts(2:end)',ones(n_basis,1))).*repmat(phi(x),n+1,1));
    funcv2(k-burn_in+1,:) = func2(zv);
end
sh = plot(zv,funcv2,'k');
hold on
th = plot(zv,f_true(zv),'--g','linewidth',2);
dh = plot(z,zeta,'r.','markersize',30);
axis(axv)
title('Fully Bayesian learning with discontinuity points')
xlabel('z')
ylabel('f(z)')
legend([sh(1) dh th],'Samples from posterior','Data points','True function')
drawnow


% Plot the ML estimate from learning 3
subplot(223)
func3 = @(x) model3.A*phi(x);
eh = plot(zv,func3(zv),'k');
hold on
th = plot(zv,f_true(zv),'--g','linewidth',2);
dh = plot(z,zeta,'r.','markersize',30);
axis(axv)
title('Maximum Likelihood learning with regularization (no discontinuity points)')
xlabel('z')
ylabel('f(z)')
legend([eh dh th],'Regularized maximum likelihood estimate','Data points','True function')
drawnow


% Plot the ML estimate from learning 4
subplot(224)
func4 = @(x) model4.A*phi(x);
eh = plot(zv,func4(zv),'k');
hold on
th = plot(zv,f_true(zv),'--g','linewidth',2);
dh = plot(z,zeta,'r.','markersize',30);
axis(axv)
title('Maximum Likelihood learning without regularization (no discontinuity points)')
xlabel('z')
ylabel('f(z)')
legend([eh dh th],'Maximum likelihood estimate','Data points','True function')