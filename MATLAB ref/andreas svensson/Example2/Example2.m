% Andreas Svensson, 2016
clear, close all

%% Generate data
f_true = @(x,u) [(x(1)/(1+x(1)^2))*sin(x(2)); x(2)*cos(x(2)) + x(1)*exp(-(x(1)^2+x(2)^2)/8) + u^3/(1+u^2+0.5*cos(x(1)+x(2)))];
g_true = @(x) x(1)/(1+0.5*sin(x(2))) + x(2)/(1+0.5*sin(x(1)));

R = 0.1; % Output noise

T = 2000; % Number of dat points
u = 2.5-5*rand(1,T); % input
y = zeros(1,T); % allocation

xt = [0;0];
for t = 1:T
    y(t) = g_true(xt) + mvnrnd(0,R);
    xt = f_true(xt,u(t));
end

%% Setup

data = iddata(y',u',1);
init_sys = n4sid(data,2); % Find linear model

[iA,iB,iC] = obsvf(init_sys.a,init_sys.b,init_sys.c);
iB = iB.*iC(2);
iC(2) = 1; % Write as state space model

g_i = @(x,u) [0 1]*x; R = 0.1; % Linear, fix observation function
nx = 2; nu = 1; ny = 1;

% Parameters for the learning
K = 500; N = 30;

% Basis functions for f
n_basis_u = 7;
n_basis_x1 = 7;
n_basis_x2 = 7;
L = zeros(1,1,nx+nu); L(:) = [5 5 5];

n_basis = n_basis_u*n_basis_x1*n_basis_x2;
jv = zeros(n_basis,1,(nx+nu));
lambda = zeros(n_basis,(nx+nu));

% 3D
for i = 1:n_basis_u
    for j = 1:n_basis_x1
        for k = 1:n_basis_x2
            ind = n_basis_x1*n_basis_x2*(i-1) + n_basis_x2*(j-1) + k;
            jv(ind,1,:) = [i,j,k];
            lambda(ind,:) = (pi.*[i,j,k]'./(2*squeeze(L))).^2;
        end
    end
end

phi = @(x,u)  prod(bsxfun(@times,L.^(-1/2),sin(pi*bsxfun(@times,bsxfun(@times,jv,(permute([u; x],[3 2 1])+L(:,ones(1,size(x,2)),:))),1./(2*L)))),3);

% GP prior:
S_SE = @(w,ell) sqrt(2*pi.*ell.^2).*exp(-(w.^2./2).*ell.^2);
V = 1000*diag(prod(S_SE(sqrt(lambda),1),2));

% Priors for Q
lQ = 100; LambdaQ = 1*eye(nx);

% Pre-allocate and initialization
model = cell(K,1);
model{1}.A = zeros(nx,n_basis);
model{1}.Q = eye(nx);


%% Run learning

tic

x_prim = zeros(nx,1,T);

for k = 1:K
    
    A = model{k}.A; Q = model{k}.Q;

    f_i = @(x,u) iA*x + iB*u(:,ones(1,size(x,2))) + ...
    A*phi(x,u(:,ones(1,size(x,2))));
    
    Q_chol = chol(Q);
    
    % Pre-allocate
    w = zeros(T,N); x_pf = zeros(nx,N,T);
    a = zeros(T,N);

    % Initialize
    if k > 1; x_pf(:,end,:) = x_prim; end
    w(1,:) = 1; w(1,:) = w(1,:)./sum(w(1,:));
    
    % CPF with ancestor sampling
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
        log_w = -(g_i(x_pf(:,:,t),u(t)) - y(t)).^2/2/R; 
        w(t,:) = exp(log_w - max(log_w));
        w(t,:) = w(t,:)/sum(w(t,:));
    end
    

    % Sample trajectory to condition on
    star = systematic_resampling(w(end,:),1);
    x_prim(:,1,T) = x_pf(:,star,T);
    for t = T-1:-1:1
        star = a(t+1,star);
        x_prim(:,1,t) = x_pf(:,star,t);
    end
    
    display(['Sampling. k = ',num2str(k), '/', num2str(K)])
    
	% Compute statistics
    linear_part = iA*squeeze(x_prim(:,1,1:T-1)) + iB*u(1:T-1);
    zeta = squeeze(x_prim(:,1,2:T)) - linear_part;
    z = phi(squeeze(x_prim(:,1,1:T-1)),u(1:T-1));
    Phi = zeta*zeta'; Psi = zeta*z'; Sig = z*z';
        
    model{k+1} = gibbs_param( Phi, Psi, Sig, V, LambdaQ,lQ,T-1);
end

time_sample = toc;
%% Evaluate

% Remove burn-in
burn_in = min(floor(1*K/4),2000);
Kb = K-burn_in;

% Generate test data
T_test = 500;
u_test = sin(2*pi*(1:T_test)/10) + sin(2*pi*(1:T_test)/25);
y_test = zeros(T_test,1);

xt = [0;0];
for t = 1:T_test
    y_test(t) = g_true(xt) + 0*mvnrnd(0,R);
    xt = f_true(xt,u_test(t));
end

% Perform test by simulating the learned model Kn times
Kn = 5;
x_test_sim = zeros(nx,T_test+1,Kb*Kn);
y_test_sim = zeros(T_test,1,Kb*Kn);

for k = 1:Kb
    A = model{k+burn_in}.A; Q = model{k+burn_in}.Q;
    f_r = @(x,u) iA*x + iB*u(:,ones(1,size(x,2))) + A*phi(x,u(:,ones(1,size(x,2))));
    g_r = g_i;
    for kn = 1:Kn
        ki = (k-1)*Kn + kn;
        for t = 1:T_test
            x_test_sim(:,t+1,ki) = f_r(x_test_sim(:,t,ki),u_test(t)) + mvnrnd(zeros(1,nx),Q)';
            y_test_sim(t,1,ki) = g_r(x_test_sim(:,t,ki)) + mvnrnd(0,R)';
        end
    end
    display(['Evaluating. k = ',num2str(k), '/', num2str(Kb)])
end
%
y_test_sim_med = median(y_test_sim,3);
y_test_sim_09 = quantile(y_test_sim,0.9,3);
y_test_sim_01 = quantile(y_test_sim,0.1,3);

% Compare to linear model
x_l = [0;0];
y_l = zeros(1,T_test);

for t = 1:T_test
    y_l(t) = iC*x_l;
    x_l = iA*x_l + iB*u_test(t);
end

% Plot
figure(4); clf
subplot(3,1,1:2)
fh = fill([1:T_test,flip(1:T_test,2)]',[y_test_sim_09; flip(y_test_sim_01,1)],0.8*[1 1 1],'linestyle','none'); hold on
lth = plot(y_test(1:T_test),'linewidth',2);
ls = plot(y_test_sim_med);
llin = plot(y_l,'k:');
legend([lth ls fh llin],'true','simulated','90% credibility interval','linear model')
title('Simulation')

subplot(3,1,3)
plot(1:T_test,abs(y_test_sim_med-y_test))
title('Error')

% Compute performance measures

y_mean = mean(y_test);
rmse = sqrt(mean((y_test_sim_med-y_test).^2));
rmse_lin = sqrt(mean((y_l'-y_test).^2));
rmsy = sqrt(mean((y_test-y_mean).^2));

display(['RMSE: ', num2str(rmse), '(',num2str(100*(1-rmse/rmsy)),'%).',' Linear: ', num2str(rmse_lin),'(',num2str(100*(1-rmse_lin/rmsy)),'%)'])

time_tot = toc;