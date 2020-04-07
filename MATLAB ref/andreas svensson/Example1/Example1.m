% Andreas Svensson, 2016

clear
% rng(100) used in the paper
%% Define experiment
T = 40;

x = zeros(1,T);
y = zeros(T,1);
x(:,1) = mvnrnd(0,10);
f_t = @(x) 10*sinc(x/7);
g = @(x) x;

Q = 4;
R = 4;

for t = 1:T
    x(:,t+1) = f_t(x(:,t)) + mvnrnd(0,Q);
    y(t) = g(x(:,t)) + mvnrnd(0,R);
end

%% Repeat the example by Tobar et al

kernel = @(xi,xj) exp(-(xi-xj).^2/10);

Nk = 8;
sv = [ 3.3185    7.5858   -2.7754    9.9696   -7.0224    5.3881   -0.1520   15.8296];

A = randn(1,Nk);
phi = @(x) A*kernel(repmat(sv',[1 size(x,2)]),repmat(x,[Nk 1]));

xv = -30:0.5:30;
%%

phiv = @(x,u) kernel(repmat(sv',[1 size(x,2)]),repmat(x,[Nk 1]));
f_g = @(x,u,A) A*phiv(x,u);
g   = @(x,u) x;

nx = 1;

prior_V = 1/0.2*exp(-0.2*bsxfun(@minus,sv,sv').^2);%0.2*eye(Nk);

reg_p = 7;
prior_pdf = @(theta) mvnpdf(theta',zeros(size(theta')),prior_V); %1; %prod(unifpdf(theta,-reg_p,reg_p));
% prior_pdf = @(theta) prod(unifpdf(theta,-reg_p,reg_p));
prior_sample = @(n) 0*unifrnd(-reg_p,reg_p,[n Nk])';

N = 40;
K_pmmh = 400; K_pgas = 400; K_pgas_GP = 400;


tic
theta_pmmh = pmmh( K_pmmh, N, Nk, zeros(T,1), y, f_g, g, nx, prior_sample, prior_pdf, Q, R);
pmmh_time = toc;
tic
theta_pgas = pgas( K_pgas, N, Nk, zeros(T,1), y, phiv, g, nx, prior_sample, prior_V/Q, Q, R);
pgas_time = toc;

%%
Nbi = 20;
prior_sample = @(n) zeros(size(1,n));
jv = (1:Nbi)'; %(-Nbi:Nbi)';
Nb = length(jv);
L = 35;
ell = 3;
lambda = @(j) (pi*j/(2*L)).^2;
S_SE = @(w,ell) 50*sqrt(2*pi.*ell.^2).*exp(-(w.^2./2).*ell.^2);
prior_V_GP = diag(S_SE(sqrt(lambda(jv)),ell));

phiv_GP = @(x,u) L^(-1/2)*sin(pi*bsxfun(@times,jv,repmat((x+L),Nb,1))./(2*L));

tic
theta_pgas_GP = pgas( K_pgas_GP, N, Nb, zeros(T,1), y, phiv_GP, g, nx, prior_sample, prior_V_GP/Q, Q, R);
pgas_GP_time = toc;
%%
gamma = zeros(1,K_pgas_GP);gamma(1:20) = 1; gamma(21:50) = 0.9;
gamma(51:end) = 0.9*(((0:K_pgas_GP-51)+1)/1).^(-0.7);

tic
theta_psaem = psaem( K_pgas_GP, N, Nb, zeros(T,1), y, phiv_GP, g, nx, prior_sample, prior_V_GP/Q, Q, R, gamma);
psaem_time = toc;

%%
Nbi = 40;
prior_sample = @(n) zeros(size(1,n));
jv = (1:Nbi)';
Nbnr = length(jv);
L = 35;

phiv_nr = @(x,u) L^(-1/2)*sin(pi*bsxfun(@times,jv,repmat((x+L),Nbnr,1))./(2*L));

tic
theta_psaem_nr = psaem( K_pgas_GP, N, Nbnr, zeros(T,1), y, phiv_nr, g, nx, prior_sample, diag(Inf*ones(Nbnr,1)), Q, R, gamma);
psaem_time_nr = toc;

%%
close all

burn_in_pmmh = 200;
burn_in_pgas = 50;

figure(5)
theta_pmmh_std = std(theta_pmmh(:,burn_in_pmmh:end),[],2);
theta_pmmh_mean = mean(theta_pmmh(:,burn_in_pmmh:end),2);
hold on
fill([xv,flip(xv)],[f_g(xv,[],theta_pmmh_mean'+theta_pmmh_std'),flip(f_g(xv,[],theta_pmmh_mean'-theta_pmmh_std'))],0.7*[1 1 1],'EdgeColor','none');
plot(xv,f_g(xv,[],theta_pmmh_mean'),'b','linewidth',2)
plot(xv,f_t(xv),'k','linewidth',2)
hold on
plot(x(1:T-1),x(2:T),'.r','markersize',7)
ylim([-10 15])
title(['Tobar et al. (',num2str(pmmh_time),' s)'])
box('on')
xlabel('$x_t$')
ylabel('$x_{t+1}$')

figure(4)
theta_pgas_std = std(theta_pgas(:,burn_in_pgas:end),[],2);
theta_pgas_mean = mean(theta_pgas(:,burn_in_pgas:end),2);
hold on
fill([xv,flip(xv)],[f_g(xv,[],theta_pgas_mean'+theta_pgas_std'),flip(f_g(xv,[],theta_pgas_mean'-theta_pgas_std'))],0.7*[1 1 1],'EdgeColor','none');
plot(xv,f_g(xv,[],theta_pgas_mean'),'b','linewidth',2)
plot(xv,f_t(xv),'k','linewidth',2)
hold on
plot(x(1:T-1),x(2:T),'.r','markersize',7)
ylim([-10 15])
title(['Same model as TObar et al, but PGAS for learning (',num2str(pgas_time),' s)'])
box('on')
xlabel('$x_t$')
ylabel('$x_{t+1}$')
%
figure(3)

covA = cov(squeeze(theta_pgas_GP(:,burn_in_pgas:end))');
meanA = mean(theta_pgas_GP(:,burn_in_pgas:end),2)';

f_m = @(x) meanA*phiv_GP(x,0);
f_std = @(x) sqrt(diag(phiv_GP(x,0)'*covA*phiv_GP(x,0)))';

% Uncertainty
fill([xv,flip(xv)],[f_m(xv)+f_std(xv), ...
                flip(f_m(xv)-f_std(xv))],0.7*[1 1 1],'EdgeColor','none');
hold on
plot(xv,f_m(xv),'b','linewidth',2)
plot(xv,f_t(xv),'k','linewidth',2)
hold on
plot(x(1:T-1),x(2:T),'.r','markersize',7)
ylim([-10 15])
title(['Our proposed model; Bayesian learning (',num2str(pgas_GP_time),' s)'])
box('on')
xlabel('$x_t$')
ylabel('$x_{t+1}$')
%

figure(2)

f_p = @(x) theta_psaem(:,end)'*phiv_GP(x,0);

hold on
plot(xv,f_p(xv),'b','linewidth',2)
plot(xv,f_t(xv),'k','linewidth',2)
hold on
plot(x(1:T-1),x(2:T),'.r','markersize',7)
ylim([-10 15])
phiv_GPv = phiv_GP(xv,0);
box('on')
title(['Our proposed model; regularized ML learning (',num2str(psaem_time),' s)'])
xlabel('$x_t$')
ylabel('$x_{t+1}$')
%

figure(1)

f_p = @(x) theta_psaem_nr(:,end)'*phiv_nr(x,0);
f_pv = f_p(xv);

plot(xv,max(min(f_pv,50),-50),'b','linewidth',2)
hold on
plot(xv,f_t(xv),'k','linewidth',2)
plot(x(1:T-1),x(2:T),'.r','markersize',7)
ylim([-10 15])
title(['Our proposed model, but without regularization (',num2str(psaem_time_nr),' s)'])
xlabel('$x_t$')
ylabel('$x_{t+1}$')
