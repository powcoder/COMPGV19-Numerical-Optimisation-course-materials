%% *COMPGV19: Tutorial 9*
%
% Marta Betcke and Kiko Rul·lan
% 
% This is an adaptation of the problem in Exrecise 1 to images 
% to demonstrate how to deal with operators on images.

%==================================================
% SETUP
%==================================================
% Set path to ISTA
addpath ISTA

clear all; 
close all;
% Set states for repeatable experiments
rng(999);
%%

%==================================================
% SIGNAL DEFINITION
%==================================================

% Image size n x n
n = 128;
N = n^2;
% Number of spikes in the signal
T = ceil(0.01*N);
% Number of observations to make
K = ceil(log(N)*T);
% Noise standard deviation
sigma = 0.005;

% Random +/- 1 signal
Itrue = zeros(n,n);
q = randperm(N);
Itrue(q(1:T)) = sign(randn(T,1));
% Plot signal
figure;
imagesc(Itrue); title('true x'); axis image
xtrue = Itrue(:);
%%

%==========================================================================================
%========================                                        ==========================
%========================          SOLVERS (comment in/out)       ==========================
%========================                                        ==========================
%==========================================================================================

%==================================================
% OPERATOR DEFINITION (I): Random orthogonal KxN measurement matrix
%==================================================
Am = randn(K,N);
Am = orth(Am')';
A = @(x) Am*x;
At = @(x) Am'*x;
%%

% %==================================================
% % OPERATOR DEFINITION (II): Welsh-Hadamard transform
% %==================================================
% 
% % 2D WH transform as a map from a vectorised image to a vectorized data
% fwht2 = @(x) reshape(fwht(fwht(reshape(x,n,n))'),[],1);
% ifwht2 = @(x) reshape(ifwht(ifwht(reshape(x,n,n))'),[],1);
% 
% WH = @(x) fwht2(x);
% WHt = @(x) ifwht2(x); % unitary formulation WH^(-1) = WH^T
% % Subsampling operator
% ind = randperm(N); 
% ind = ind(1:K);
% S = @(x) x(ind);
% St = @(x) subsampleAdjoint(x,N,ind);
% % Subsampled WH transform as a map from a vectorised image to a vectorized data
% A = @(x) S(WH(x)); 
% At = @(x) WHt(St(x));


%==================================================
% GENERATE NOISY MEASUREMENTS
%==================================================
y = A(xtrue);
%yn = y.*(1 + sigma*randn(K,1));         % multiplicative 0-mean Gaussian noise
yn = y + sigma*max(abs(y))*randn(K, 1);  % additive 0-mean Gaussian noise
%%

%%
%==================================================
% SOLVE USING L1-MAGIC 
%==================================================
% This are methods we talked about in the lecture but the reformulation of $\ell_1$ norm is new.
% Still it should be interesting to see how they perform for this problem as a comparison 
% to the dedicated optimised solvers.

%====================
% Equality constraints (no noise)
%====================
% Initial guess: min L2-norm (energy) solution
x0 = At(y);
% Compute and plot Primal-Dual iterations
xPD = l1eq_pd(x0, A, At, y, 1e-3, 30, 1e-8, 200);
figure;
subplot(1,2,1); imagesc(reshape(xPD,n,n)); title('L1 Magic - Primal Dual'); axis image; colorbar
subplot(1,2,2); imagesc(reshape(xPD - xtrue,n,n)); title('Error: L1 Magic - Primal Dual'); axis image; colorbar
%

%====================
% Inequality constraints - take epsilon a little bigger than sigma*sqrt(K)
%====================
% Initial guess: min L2-norm (energy) solution
x0 = At(yn);
epsilon =  sigma*sqrt(K)*sqrt(1 + 2*sqrt(2)/sqrt(K));  
% Compute and plot LogBarrier iterations
xLOGB = l1qc_logbarrier(x0, A, At, yn, epsilon, 1e-3, 50, 1e-8, 500);
figure,
subplot(1,2,1); imagesc(reshape(xLOGB,n,n)); title('L1 Magic - Log Barrier'); axis image; colorbar
subplot(1,2,2); imagesc(reshape(xLOGB - xtrue,n,n)); title('Error: L1 Magic - Log Barrier'); axis image; colorbar
%%



%==================================================
% Initialization ISTA/FISTA ADMM
%==================================================
% Initial guess: 
x0 = At(yn);  % min L2-norm (energy) solution
%x0 = zeros(N,1); % all 0s

% Regularization parameter
lambda = 0.01*max(abs(At(yn)));
%%

%==================================================
% FUNCTION DEFINITION ISTA/FISTA
%==================================================
% L2 Data fit term 
F.f = @(x,y) 0.5*sum((A(x) - y).^2); 
% Gradient of the data fit term
F.df = @(x,y) At(A(x) - y); 

% L1-Penalty function
G.f = @(x) norm(x, 1); 
% Derivative of the penalty function (subgradient)
G.df = @(x) sign(x);  
% Proximal operator wrt 1-norm || x ||_1
G.prox = @(x, alpha) softThresh(x, alpha); 

% Objective function as in eg gradient descent 
obj.f = @(x,lambda,y) F.f(x,y) + lambda*G.f(x); % Regularized objective function
obj.df = @(x,lambda,y) F.df(x,y) + lambda*G.df(x); % Subgradient of obj
obj.L = 1; % Lipschitz constant for the gradient
%%

%==================================================
% SOLVE USING FISTA
%==================================================
       
%==================== 
% Gradient descent
%==================== 
[xGD, infoGD.obj, infoGD.mse] = gradientDescent(yn, obj, lambda, x0, 1, 5e-4, 100, xtrue);
figure;
subplot(1,2,1); imagesc(reshape(xGD,n,n)); title('GradientDescent'); axis image; colorbar
subplot(1,2,2); imagesc(reshape(xGD - xtrue,n,n)); title('Error: GradientDescent'); axis image; colorbar
%%

%==================== 
% ISTA 
%==================== 
[xISTA, infoISTA.obj, infoISTA.mse] = ista(yn, F, G, lambda, x0, 1, 5e-4, 500, xtrue);
figure;
subplot(1,2,1); imagesc(reshape(xISTA,n,n)); title('ISTA'); axis image; colorbar
subplot(1,2,2); imagesc(reshape(xISTA - xtrue,n,n)); title('Error: ISTA'); axis image; colorbar

%%

%==================== 
% FISTA: ISTA + Nesterov acceleration
%==================== 
[xNEST, infoNEST.obj, infoNEST.mse] = fista(yn, F, G, lambda, x0, 1, 5e-4, 100, xtrue);
subplot(1,2,1); imagesc(reshape(xNEST,n,n)); title('FISTA'); axis image; colorbar
subplot(1,2,2); imagesc(reshape(xNEST - xtrue,n,n)); title('Error: FISTA'); axis image; colorbar

%%

%==================================================
% SOLVE USING ADMM
%==================================================
% Initialisation
E = @(x) x;
F = @(x) -x;
b = 0*x0;
Proxy = @(x, rho) softThresh(x, lambda*rho);
% Solve for u (first version of primal variable split u = v) 
%   argmin || A u - y ||_2^2 + rho || u - q ||_2^2
% is equivalent to solution of NE
%   u = (A^T A + rho I)^(-1) (A^T y + rho q)
% Here A has a special structure, SWM formula can be used to simplify 
%   (A^T A + rho I)^(-1) = 1/rho * ( I - 1/(rho+1)* A^T A )
invLS = @(A, Atr,f,rho) 1/rho*( f  - 1/(rho+1)*Atr(A(f)) );
para.stopTolerance = 1e-6;
para.maxIter = 200;
para.rho = 1e-3;
% ADMM with strucutred solve for || A u - y ||_2^2 + rho || u - q ||_2^2.
% To use generic itertive solver replave invLS with []
[xADMM, v, w, iter, stopValue, uvIterates, info] = ADMM(yn, A, At, invLS, E, E, F, b, Proxy, x0, para);
% Plot results
figure;
subplot(1,2,1); imagesc(reshape(xADMM,n,n)); title('ADMM'); axis image; colorbar
subplot(1,2,2); imagesc(reshape(xADMM - xtrue,n,n)); title('Error: ADMM'); axis image; colorbar
%%


