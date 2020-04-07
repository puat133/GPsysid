% Andreas Svensson, 2016
clear, close all
load dataBenchmark % Available here: http://homepages.vub.ac.be/~mschouke/benchmarkCascadedTanks.html
u = uEst';
y = yEst';

rng(1) % used in the paper

% Center data around 0
u_ofs = mean(u);
y_ofs = mean(y);
T = length(u);
u = u(1:T) - u_ofs;
y = y(1:T) - y_ofs;

% find linear model
data = iddata(y',u',Ts);
init_sys = n4sid(data,2);

[iA,iB,iC] = obsvf(init_sys.a,init_sys.b,init_sys.c);
iB = iB.*iC(2);
iC(2) = 1;

%
tic

% Model:
% Nonlinear model for f_1(x_1,u) (disc in x_1) and f_2(x_1,x_2,u) (disc in x_2), known g_x. 2 dimensions.

g_i = @(x,u) [0 1]*x; R = 0.01;
nx = 2; nu = 1; ny = 1;

% Parameters for the algorithm, priors, and basis functions
K = 10000; N = 30;

% Basis functions for f:
n_basis_u = 5;
n_basis_x1 = 5;
n_basis_x2 = 5;
L = zeros(1,1,nx+nu); L(:) = [5 15 6];

n_basis_1 = n_basis_u*n_basis_x1;
jv_1 = zeros(n_basis_1,1,nx-1+nu);
lambda_1 = zeros(n_basis_1,nx-1+nu);

n_basis_2 = n_basis_u*n_basis_x1*n_basis_x2;
jv_2 = zeros(n_basis_2,1,(nx+nu));
lambda_2 = zeros(n_basis_2,(nx+nu));

% 2D (f_1)
for i = 1:n_basis_u
    for j = 1:n_basis_x1
        ind = n_basis_x1*(i-1) + j;
        jv_1(ind,1,:) = [i,j];
        lambda_1(ind,:) = (pi.*[i,j]'./(2*squeeze(L([1 2])))).^2;
    end
end

% 3D (f_2)
for i = 1:n_basis_u
    for j = 1:n_basis_x1
        for k = 1:n_basis_x2
            ind = n_basis_x1*n_basis_x2*(i-1) + n_basis_x2*(j-1) + k;
            jv_2(ind,1,:) = [i,j,k];
            lambda_2(ind,:) = (pi.*[i,j,k]'./(2*squeeze(L))).^2;
        end
    end
end

phi_1 = @(x1,u) prod(bsxfun(@times,L(:,:,[1 2]).^(-1/2),sin(pi*bsxfun(@times,bsxfun(@times,jv_1,(permute([u;x1],[3 2 1])+L(:,ones(1,size(x1,2)),[1 2]))),1./(2*L(:,:,[1 2]))))),3);
phi_2 = @(x,u)  prod(bsxfun(@times,L.^(-1/2),sin(pi*bsxfun(@times,bsxfun(@times,jv_2,(permute([u; x],[3 2 1])+L(:,ones(1,size(x,2)),:))),1./(2*L)))),3);

% GP prior:
S_SE = @(w,ell) sqrt(2*pi.*ell.^2).*exp(-(w.^2./2).*ell.^2);
V1 = @(n1) 100*diag(repmat(prod(S_SE(sqrt(lambda_1),repmat([3 3]  ,[n_basis_1,1])),2),[n1+1 1]));
V2 = @(n2) 100*diag(repmat(prod(S_SE(sqrt(lambda_2),repmat([3 3 3],[n_basis_2,1])),2),[n2+1 1]));

% Priors for Q
lQ1 = 1000; LambdaQ1 = 1*eye(1);
lQ2 = 1000; LambdaQ2 = 1*eye(1);

% Pre-allocate and initialization
model_state_1 = cell(K,1);
model_state_2 = cell(K,1);

model_state_1{1}.A = zeros(1,n_basis_1);
model_state_1{1}.Q = 1;
model_state_1{1}.n = 0;
model_state_1{1}.pts = [-L(2) L(2)];

model_state_2{1}.A = zeros(1,2*n_basis_2);
model_state_2{1}.Q = 1;
model_state_2{1}.n = 1;
model_state_2{1}.pts = [-L(3) 4.4 L(3)];

% Priors for discontinuity points

p1 = 0.9;

% Sanity check by sampling from prior (no discontinuity points for simplicitly)

figure(1), clf
for i = 1:3

    model_state_1p = gibbs_param(0, 0, 0, V1(0), LambdaQ1,lQ1,0);
    model_state_2p = gibbs_param(0, 0, 0, V2(0), LambdaQ2,lQ2,0);

    f_i = @(x,u) [iA(1,:)*x + iB(1)*u + model_state_1p.A*phi_1(x(1,:,:),u); iA(2,:)*x + iB(2)*u + model_state_2p.A*phi_2(x,u)];

    ys = zeros(1,T);
    xs = [0;0];

    for t = 1:T
        ys(t) = g_i(xs);
        xs = f_i(xs,u(t));
    end

    plot(ys)
    hold on
    drawnow
end


%% Run learning algorithm

% Pre-allocate
x_prim = zeros(nx,1,T);

% Run MCMC algorithm!
for k = 1:K
    
    Qi = zeros(nx);
    
    pts1 = model_state_1{k}.pts; n1 = model_state_1{k}.n; Ai1 = model_state_1{k}.A; Qi(1,1) = model_state_1{k}.Q;
    pts2 = model_state_2{k}.pts; n2 = model_state_2{k}.n; Ai2 = model_state_2{k}.A; Qi(2,2) = model_state_2{k}.Q;
    
    f_i = @(x,u) iA*x + iB*u(:,ones(1,size(x,2))) + ...
    [Ai1*(bsxfun(@ge,repmat(x(1,:),[n_basis_1*(n1+1) 1]),kron(pts1(1:end-1)',ones(n_basis_1,1))).*bsxfun(@lt,repmat(x(1,:),[n_basis_1*(n1+1) 1]),kron(pts1(2:end)',ones(n_basis_1,1))).*repmat(phi_1(x(1,:),u(:,ones(1,size(x,2)))),n1+1,1));
     Ai2*(bsxfun(@ge,repmat(x(2,:),[n_basis_2*(n2+1) 1]),kron(pts2(1:end-1)',ones(n_basis_2,1))).*bsxfun(@lt,repmat(x(2,:),[n_basis_2*(n2+1) 1]),kron(pts2(2:end)',ones(n_basis_2,1))).*repmat(phi_2(x,u(:,ones(1,size(x,2)))),n2+1,1))];
    
    Q_chol = chol(Qi);
    
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

                waN = w(t-1,:).*mvnpdf(f_i(x_pf(:,:,t-1),u(t-1))',x_pf(:,N,t)',Qi)';
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

    zeta1 = squeeze(x_prim(1,1,2:T))' - linear_part(1,:);
    z1 = permute(phi_1(squeeze(x_prim(1,1,1:T-1))',u(1:T-1)),[1 3 2]);
    zx1 = squeeze(x_prim(1,1,1:T-1))';
    zu1 = u(1:T-1);
    
    zeta2 = squeeze(x_prim(2,1,2:T))' - linear_part(2,:);
    z2 = permute(phi_2(squeeze(x_prim(:,1,1:T-1)),u(1:T-1)),[1 3 2]);
    zx2 = squeeze(x_prim(:,1,1:T-1));
    zu2 = u(1:T-1);
    
    % Propose a new jump model
    n1 = geornd(p1); pts1 = sort([-L(2)*100, L(2)-2*L(2)*rand([1 n1]), L(2)*100]);
    
    % Compute its statistics and marginal likelihood
    zp1 = bsxfun(@ge,repmat(zx1,[n_basis_1*(n1+1) 1]),kron(pts1(1:end-1)',ones(n_basis_1,1))).*bsxfun(@lt,repmat(zx1,[n_basis_1*(n1+1) 1]),kron(pts1(2:end)',ones(n_basis_1,1))).*repmat(phi_1(zx1,zu1),n1+1,1);
    prop1.Phi = zeta1*zeta1'; prop1.Psi = zeta1*zp1'; prop1.Sig = zp1*zp1'; prop1.V = V1(n1);
    prop1.marginal_likelihood = compute_marginal_likelihood(prop1.Phi,prop1.Psi,prop1.Sig,prop1.V,LambdaQ1,lQ1,T-1);
    prop1.n = n1; prop1.pts = pts1;
    
    if k > 1
        % Alternatively staying with the current jump model
        n1 = model_state_1{k}.n; pts1 = model_state_1{k}.pts;

        % Compute its statistics and marginal likelihood
        zp1 = bsxfun(@ge,repmat(zx1,[n_basis_1*(n1+1) 1]),kron(pts1(1:end-1)',ones(n_basis_1,1))).*bsxfun(@lt,repmat(zx1,[n_basis_1*(n1+1) 1]),kron(pts1(2:end)',ones(n_basis_1,1))).*repmat(phi_1(zx1,zu1),n1+1,1);
        curr1.Phi = zeta1*zeta1'; curr1.Psi = zeta1*zp1'; curr1.Sig = zp1*zp1'; curr1.V = V1(n1);
        curr1.marginal_likelihood = compute_marginal_likelihood(curr1.Phi,curr1.Psi,curr1.Sig,curr1.V,LambdaQ1,lQ1,T-1);
        curr1.n = n1; curr1.pts = pts1;
    end
    
    dv = rand;
    if (k == 1) || (dv < min(exp(prop1.marginal_likelihood - curr1.marginal_likelihood),1))
        jmodel = prop1;
        accept1 = 1*(jmodel.n~=model_state_1{k}.n);
    else
        jmodel = curr1;
        accept1 = 0;
    end
        
    model_state_1{k+1} = gibbs_param( jmodel.Phi, jmodel.Psi, jmodel.Sig, jmodel.V, LambdaQ1,lQ1,T-1);
    model_state_1{k+1}.n = jmodel.n; model_state_1{k+1}.pts = jmodel.pts;

    if accept1 > 0
        display(['Accept dim 1! New n1 is ', num2str(jmodel.n),'.'])
    end
    
    % Fixed discontinutiy point in f_2
    zp2 = bsxfun(@ge,repmat(zx2(2,:),[n_basis_2*(n2+1) 1]),kron(pts2(1:end-1)',ones(n_basis_2,1))).*bsxfun(@lt,repmat(zx2(2,:),[n_basis_2*(n2+1) 1]),kron(pts2(2:end)',ones(n_basis_2,1))).*repmat(phi_2(zx2,zu2),n2+1,1);
    Phi2 = zeta2*zeta2'; Psi2 = zeta2*zp2'; Sig2 = zp2*zp2';
        
    model_state_2{k+1} = gibbs_param( Phi2, Psi2, Sig2, V2(1), LambdaQ2,lQ2,T-1);
    model_state_2{k+1}.n = n2; model_state_2{k+1}.pts = pts2;
end

time_sample = toc;
%% Test

% Remove burn-in
burn_in = min(floor(1*K/4),2000);
Kb = K-burn_in;

% Center test data around same working point as training data
u_test = uVal - u_ofs;
y_test = yVal - y_ofs;
T_test = length(u_test);

data_test = iddata(y_test,u_test,Ts);

Kn = 2;
x_test_sim = zeros(nx,T_test+1,Kb*Kn);
y_test_sim = zeros(T_test,1,Kb*Kn);

for k = 1:Kb
    Qr = zeros(nx);
    pts1 = model_state_1{k+burn_in}.pts; n1 = model_state_1{k+burn_in}.n; Ar1 = model_state_1{k+burn_in}.A; Qr(1,1) = model_state_1{k+burn_in}.Q;
    pts2 = model_state_2{k+burn_in}.pts; n2 = model_state_2{k+burn_in}.n; Ar2 = model_state_2{k+burn_in}.A; Qr(2,2) = model_state_2{k+burn_in}.Q;
    f_r = @(x,u) iA*x + iB*u(:,ones(1,size(x,2))) + ...
    [Ar1*(bsxfun(@ge,repmat(x(1,:),[n_basis_1*(n1+1) 1]),kron(pts1(1:end-1)',ones(n_basis_1,1))).*bsxfun(@lt,repmat(x(1,:),[n_basis_1*(n1+1) 1]),kron(pts1(2:end)',ones(n_basis_1,1))).*repmat(phi_1(x(1,:),u(:,ones(1,size(x,2)))),n1+1,1));
     Ar2*(bsxfun(@ge,repmat(x(2,:),[n_basis_2*(n2+1) 1]),kron(pts2(1:end-1)',ones(n_basis_2,1))).*bsxfun(@lt,repmat(x(2,:),[n_basis_2*(n2+1) 1]),kron(pts2(2:end)',ones(n_basis_2,1))).*repmat(phi_2(x,u(:,ones(1,size(x,2)))),n2+1,1))];
    g_r = g_i;
    for kn = 1:Kn
        ki = (k-1)*Kn + kn;
        for t = 1:T_test
            x_test_sim(:,t+1,ki) = f_r(x_test_sim(:,t,ki),u_test(t)) + mvnrnd(zeros(1,nx),Qr)';
            y_test_sim(t,1,ki) = g_r(x_test_sim(:,t,ki)) + mvnrnd(0,R)';
        end
    end
    display(['Evaluating. k = ',num2str(k), '/', num2str(Kb), '. n1 = ', num2str(model_state_1{k+burn_in}.n), ', n2 = ', num2str(model_state_2{k+burn_in}.n)])
end

y_test_sim_med = median(y_test_sim,3);
y_test_sim_09 = quantile(y_test_sim,0.9,3);
y_test_sim_01 = quantile(y_test_sim,0.1,3);

% Compare to linear model
x_l = [0;0];
y_sim_l = zeros(1,T_test);

for t = 1:T
    y_sim_l(t) = iC*x_l;
    x_l = iA*x_l + iB*u_test(t);
end

%% Compare to NLARX in system identification toolbox
Options = nlarxOptions();
Narx1 = nlarx(data, [5 5 1], 'sigmoidnet',Options);
Narx2 = nlarx(data, [5 5 1], 'wavenet',Options);
y_sim_nlarx1 = compare(data_test, Narx1);
y_sim_nlarx2 = compare(data_test, Narx2);
% Options = nlarxOptions('Focus', 'simulation');
Narx1s = nlarx(data, [5 5 1], 'sigmoidnet',Options);
Narx2s = nlarx(data, [5 5 1], 'wavenet',Options);
y_sim_nlarx1s = compare(data_test, Narx1s);
y_sim_nlarx2s = compare(data_test, Narx2s);

%%
rmse_ss = sqrt(mean((y_sim_l'-y_test).^2));
rmse_nlarx1 = sqrt(mean((y_sim_nlarx1.y-y_test).^2));
rmse_nlarx2 = sqrt(mean((y_sim_nlarx2.y-y_test).^2));
rmse_nlarx1s = sqrt(mean((y_sim_nlarx1s.y-y_test).^2));
rmse_nlarx2s = sqrt(mean((y_sim_nlarx2s.y-y_test).^2));
rmse_sim = sqrt(mean((y_test_sim_med-y_test).^2));
%%
colors = [240 0 0; 255 128 0; 255 200 40; 0 121 64; 64 64 255; 160 0 192; 0 0 0]./255;

figure(2), clf
subplot(311)
tv = Ts*(0:T_test-1);
fill([tv,flip(tv,2)]',[y_test_sim_09; flip(y_test_sim_01,1)]+y_ofs,0.6*[1 1 1],'linestyle','none');
hold on
plot(tv,y_test_sim_med+y_ofs,'linewidth',1.3,'color',colors(6,:));
plot(tv,y_test+y_ofs,':','linewidth',1.3,'color',colors(7,:));

xlim([0 tv(end)])
ylim([1 12])
ylabel('output (V)')
set(gca,'xTick',0:1000:4000)

subplot(312)
tv = Ts*(0:T_test-1);
hold on
plot(tv,y_sim_l       +y_ofs,'-','linewidth',1.3,'color',colors(1,:));
plot(tv,y_sim_nlarx1.y +y_ofs,'-','linewidth',1.3,'color',colors(2,:));
plot(tv,y_sim_nlarx1s.y+y_ofs,'-','linewidth',1.3,'color',colors(4,:));
plot(tv,y_sim_nlarx2.y +y_ofs,'-','linewidth',1.3,'color',colors(3,:));
plot(tv,y_sim_nlarx2s.y+y_ofs,'-','linewidth',1.3,'color',colors(5,:));
plot(tv,y_test+y_ofs,         ':','linewidth',1.3,'color',colors(7,:));

xlim([0 tv(end)])
ylim([1 12])
xlabel('time (s)')
ylabel('output (V)')
set(gca,'xTick',0:1000:4000)
box on

subplot(313)
plot([0 1],[0 1],':','linewidth',2,'color',colors(7,:));
hold on
plot([0 1],[0 1],'-','linewidth',2,'color',colors(1,:));
plot([0 1],[0 1],'-','linewidth',2,'color',colors(2,:));
plot([0 1],[0 1],'-','linewidth',2,'color',colors(3,:));
plot([0 1],[0 1],'-','linewidth',2,'color',colors(4,:));
plot([0 1],[0 1],'-','linewidth',2,'color',colors(5,:));
plot([0 1],[0 1],'-','linewidth',2,'color',colors(6,:));
fill([0 0 1 1]',[0 1 0 1],0.6*[1 1 1],'linestyle','none');

legend('Validation data',...
    ['2nd order linear state space model. RMSE: ', num2str(rmse_ss,2)],...
    ['5th order NARX with sigmoidnet. RMSE: ', num2str(rmse_nlarx1,2)],...
    ['~~~~~~~~ " ~~~~~~~ simulation focus. RMSE: ', num2str(rmse_nlarx1s,2)],...    
    ['5th order NARX with wavelets. RMSE: ', num2str(rmse_nlarx2,2)],...
    ['~~~~~~~~ " ~~~~~~~ simulation focus. RMSE: ', num2str(rmse_nlarx2s,2)],...
    ['The proposed model. RMSE: ' , num2str(rmse_sim,2)],...
    'Credibility interval for the proposed method.','location','south')
axis([2 3 2 3])
axis off