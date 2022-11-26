%% *COMPGV19: Tutorial 9*
%
% Marta Betcke and Kiko Rul·lan
% 

% Download l1-magig from: https://statweb.stanford.edu/~candes/l1magic/ 

%==================================================
% SETUP
%==================================================
% Set path to l1-magic
addpath l1magic
addpath l1magic/Optimization
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

% Signal length
N = 2^13;
% Number of spikes in the signal
T = 100;
% Number of observations to make
K = 2^10;
% Noise standard deviation
sigma = 0.005;

% Random +/- 1 signal
xtrue = zeros(N,1);
q = randperm(N);
xtrue(q(1:T)) = sign(randn(T,1));
% Plot signal
figure;
plot(xtrue);
ylim([-1.2 1.2]);
xlabel('n');
ylabel('Amplitude');
title('X');
%saveas(gcf, 'Xtrue');
%%

%==========================================================================================
%========================                                        ==========================
%========================          SOLVERS (I)                   ==========================
%========================                                        ==========================
%==========================================================================================
strOperator = '_RandOrth';
%==================================================
% OPERATOR DEFINITION (I): Random orthogonal KxN measurement matrix
%==================================================
Am = randn(K,N);
Am = orth(Am')';
A = @(x) Am*x;
At = @(x) Am'*x;
%%

%==================================================
% GENERATE NOISY MEASUREMENTS
%==================================================
y = A(xtrue);
%yn = y.*(1 + sigma*randn(K,1));         % multiplicative 0-mean Gaussian noise
yn = y + sigma*max(abs(y))*randn(K, 1);  % additive 0-mean Gaussian noise
%%

%==================================================
% SOLVE USING L1-MAGIC 
%==================================================
% This are methods we talked about in the lecture but the reformulation of $\ell_1$ norm is new.
% Still it should be interesting to see how they perform for this problem as a comparison 
% to the dedicated optimised solvers.

% Initial guess: min L2-norm (energy) solution
x0 = At(y);
% Plot initial guess
figure;
plot(xtrue);
hold on;
plot(x0,'r--o', 'MarkerSize',10);
ylim([-1.2 1.2]);
legend('x true', 'x0 - initial guess (no noise)');
title('L1 Magic - Initial Guess');
%saveas(gcf, strcat('L1Magic-InitialGuess', strOperator));
%%

%====================
% Equality constraints (no noise)
%====================
% Compute and plot Primal-Dual iterations
xPD = l1eq_pd(x0, A, At, y, 1e-3, 30, 1e-8, 200);
figure;
plot(xtrue); 
hold on; 
plot(xPD, 'r:o', 'MarkerSize',10)
legend('x true', 'xPD');
ylim([-1.2 1.2]);
title('L1 Magic - Primal Dual');
%saveas(gcf, strcat('L1Magic-PD', strOperator));
%%

%====================
% Inequality constraints - take epsilon a little bigger than sigma*sqrt(K)
%====================
epsilon =  sigma*sqrt(K)*sqrt(1 + 2*sqrt(2)/sqrt(K));  
% Compute and plot LogBarrier iterations
xLOGB = l1qc_logbarrier(x0, A, At, yn, epsilon, 1e-3, 50, 1e-8, 500);
figure;
plot(xtrue); 
hold on;
plot(xLOGB, 'r:o', 'MarkerSize',10);
legend('x true', 'xLOGB');
ylim([-1.2 1.2]);
title('L1 Magic - Log Barrier');
%saveas(gcf, strcat('L1Magic-LogBarrier', strOperator));
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
plot(xtrue); hold on; plot(xGD, 'r:o', 'MarkerSize',10);
ylim([-1.2 1.2]);
legend('x true', 'xGD');
title('Gradient Descent');
%saveas(gcf, strcat('GradientDescent', strOperator));
%%

%==================== 
% ISTA 
%==================== 
[xISTA, infoISTA.obj, infoISTA.mse] = ista(yn, F, G, lambda, x0, 1, 5e-4, 500, xtrue);
figure;
plot(xtrue); hold on; plot(xISTA, 'r:o', 'MarkerSize',10); 
ylim([-1.2 1.2]);
legend('x true', 'xISTA');
title('ISTA');
%saveas(gcf, strcat('ISTA', strOperator));
%%

%==================== 
% FISTA: ISTA + Nesterov acceleration
%==================== 
[xNEST, infoNEST.obj, infoNEST.mse] = fista(yn, F, G, lambda, x0, 1, 5e-4, 100, xtrue);
figure;
plot(xtrue); hold on; plot(xNEST, 'r:o', 'MarkerSize',10);
ylim([-1.2 1.2]); 
legend('x true', 'xFISTA');
title('FISTA');
%saveas(gcf, strcat('FISTA', strOperator));
%%

%==================================================
% SOLVE USING ADMM
%==================================================
% Initialisation
E = @(x) x;
F = @(x) -x;
b = 0*x0;
J = @(x) lambda*G.f(x);
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
[u, v, w, iter, stopValue, uvIterates, info] = ADMM(yn, A, At, invLS, E, E, F, b, Proxy, x0, para);
% Plot results
figure;
plot(xtrue); 
hold on; 
plot(u, 'r:o', 'MarkerSize',10);
ylim([-1.2 1.2]); 
legend('x true', 'x ADMM');
title('ADMM');
%saveas(gcf, strcat('ADMM', strOperator));
%%


pause

%==========================================================================================
%========================                                        ==========================
%========================          SOLVERS (II)                  ==========================
%========================                                        ==========================
%==========================================================================================
strOperator = '_WH';
%==================================================
% OPERATOR DEFINITION (II): Welsh-Hadamard transform
%==================================================
WH = @(x) fwht(x);
WHt = @(x) ifwht(x); % unitary formulation WH^(-1) = WH^T
% Subsampling operator
ind = randperm(N); 
ind = ind(1:K);
S = @(x) x(ind);
St = @(x) subsampleAdjoint(x,N,ind);
% Subsampled WH transform
A = @(x) S(WH(x)); 
At = @(x) WHt(St(x));
%%

%==================================================
% GENERATE NOISY MEASUREMENTS
%==================================================
y = A(xtrue);
%yn = y.*(1 + sigma*randn(K,1));         % multiplicative 0-mean Gaussian noise
yn = y + sigma*max(abs(y))*randn(K, 1);  % additive 0-mean Gaussian noise
%%

%==================================================
% SOLVE USING L1-MAGIC 
%==================================================
% Initial guess: min L2-norm (energy) solution
x0 = At(y);
% Plot initial guess
figure;
plot(xtrue);
hold on;
plot(x0,'r--o', 'MarkerSize',10);
ylim([-1.2 1.2]);
legend('x true', 'x0 - initial guess (no noise)');
title('L1 Magic - Initial Guess');
%saveas(gcf, strcat('L1Magic-InitialGuess', strOperator));
%%

%====================
% Equality constraints (no noise)
%====================
% Compute and plot Primal-Dual iterations
xPD = l1eq_pd(x0, A, At, y, 1e-3, 30, 1e-8, 200);
figure;
plot(xtrue); 
hold on; 
plot(xPD, 'r:o', 'MarkerSize',10)
legend('x true', 'xPD');
ylim([-1.2 1.2]);
title('L1 Magic - Primal Dual');
%saveas(gcf, strcat('L1Magic-PD', strOperator));
%%

%%  %====================
%%  % Inequality constraints - take epsilon a little bigger than sigma*sqrt(K)
%%  %====================
%%  epsilon =  sigma*sqrt(K)*sqrt(1 + 2*sqrt(2)/sqrt(K));  
%%  % Compute and plot LogBarrier iterations
%%  xLOGB = l1qc_logbarrier(x0, A, At, yn, epsilon, 1e-3, 50, 1e-8, 500);
%%  figure;
%%  plot(xtrue); 
%%  hold on;
%%  plot(xLOGB, 'r:o', 'MarkerSize',10);
%%  legend('x true', 'xLOGB');
%%  ylim([-1.2 1.2]);
%%  title('L1 Magic - Log Barrier');
%%  %saveas(gcf, strcat('L1Magic-LogBarrier', strOperator));
%%  %%


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
clear F
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
plot(xtrue); hold on; plot(xGD, 'r:o', 'MarkerSize',10);
ylim([-1.2 1.2]);
legend('x true', 'xGD');
title('Gradient Descent');
%saveas(gcf, strcat('GradientDescent', strOperator));
%%

%==================== 
% ISTA 
%==================== 
[xISTA, infoISTA.obj, infoISTA.mse] = ista(yn, F, G, lambda, x0, 1, 5e-4, 500, xtrue);
figure;
plot(xtrue); hold on; plot(xISTA, 'r:o', 'MarkerSize',10); 
ylim([-1.2 1.2]);
legend('x true', 'xISTA');
title('ISTA');
%saveas(gcf, strcat('ISTA', strOperator));
%%

%==================== 
% FISTA: ISTA + Nesterov acceleration
%==================== 
[xNEST, infoNEST.obj, infoNEST.mse] = fista(yn, F, G, lambda, x0, 1, 5e-4, 100, xtrue);
figure;
plot(xtrue); hold on; plot(xNEST, 'r:o', 'MarkerSize',10);
ylim([-1.2 1.2]); 
legend('x true', 'xFISTA');
title('FISTA');
%saveas(gcf, strcat('FISTA', strOperator));
%%

%==================================================
% SOLVE USING ADMM
%==================================================
% Initialisation
E = @(x) x;
F = @(x) -x;
b = 0*x0;
J = @(x) lambda*G.f(x);
Proxy = @(x, rho) softThresh(x, lambda*rho);
% Solve for u (first version of primal variable split u = v) 
%   argmin || A u - y ||_2^2 + rho || u - q ||_2^2
% is equivalent to solution of NE
%   u = (A^T A + rho I)^(-1) (A^T y + rho q)
% Here A has a special structure, SWM formula can be used to simplify 
%   (A^T A + rho I)^(-1) = 1/rho * ( I - 1/(rho+1)* A^T A )
invLS = @(A, Atr,f,rho) 1/rho*( f  - 1/(rho+1)*Atr(A(f)) );
%invLS = @(y,q,rho) 1/rho*( (Atr(y) + rho*q)  - 1/(rho+1)*Atr( A (Atr(y) + rho*q) ) );
para.stopTolerance = 1e-6;
para.maxIter = 200;
para.rho = 1e-3;
% ADMM with strucutred solve for || A u - y ||_2^2 + rho || u - q ||_2^2.
% To use generic itertive solver replave invLS with []
[u, v, w, iter, stopValue, uvIterates, info] = ADMM(yn, A, At, invLS, E, E, F, b, Proxy, x0, para);
% Plot results
figure;
plot(xtrue); 
hold on; 
plot(u, 'r:o', 'MarkerSize',10);
ylim([-1.2 1.2]); 
legend('x true', 'x ADMM');
title('ADMM');
%saveas(gcf, strcat('ADMM', strOperator));
%%

%==========================================================================================
%========================                                        ==========================
%========================       INCLUDE FUNCTIONS                ==========================
%========================                                        ==========================
%==========================================================================================

%% l1eq_pd
%
% <include>l1eq_pd.m</include>
%%


%% l1qc_logbarrier
%
% <include>l1qc_logbarrier.m</include>
%%

%% ista
%
% <include>ista.m</include>
%%

%% fista
%
% <include>fista.m</include>
%%

%% ADMM
%
% <include>ADMM.m</include>
%%
