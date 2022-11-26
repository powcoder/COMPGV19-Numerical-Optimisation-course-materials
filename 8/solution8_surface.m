%% *COMPGV19: Tutorial 6*
%
% Marta Betcke and Kiko Rul·lan
% 

%%
% *Exercise 3*
% Minimise the surface given by the boundary conditions using Algorithm 7.1
close all;
%clear all;
tic;
%==============================
% Parameters
%==============================
maxIter = 1000;
tol = 1e-4;

%==============================
% Initialisation
%==============================
% Dimensions
q = 14 % dimension of interior square
n = q+2; % dimension of square with boundary
s = linspace(0,1-1/n,n-1);
% % Boundary 1
% leftBoundary   = sin(pi*s+pi);   % left boundary  
% rightBoundary  = sin(3*pi*s+pi); % right boundary 
% topBoundary    = sin(3*pi*s);    % top boundary   
% bottomBoundary = sin(pi*s);      % bottom boundary

% % Boundary 2
% leftBoundary   = 2*s;                  % left boundary  
% topBoundary    = exp(-6*s)+1;          % right boundary 
% rightBoundary  = 1+exp(6*s)/exp(6);    % top boundary    
% bottomBoundary = 2-2*s;                % bottom boundary

% Boundary 3
leftBoundary   = (2*(s-0.5)).^3 - 2*(s-0.5);    % left boundary  
topBoundary    = 0*s + (2*0.5)^3 - 1;  % right boundary 
rightBoundary  = (2*(-s+0.5)).^3 + 2*(s-0.5);    % top boundary    
bottomBoundary = 0*s + -(2*0.5)^3 + 1; % bottom boundary

bndFun = [leftBoundary topBoundary rightBoundary bottomBoundary];
figure(3), plot(bndFun); title('Boundary conditions')
% Indicies of the boundary nodes in counter clockwise order
bndInd = [1:q+2, (1:q)*(q+2) + ones(1,q)*(q+2), (q+1)*(q+2) + (q+2:-1:1), (q:-1:1)*(q+2) + ones(1,q)];
boundaryM = zeros(q+2,q+2);
boundaryM(bndInd(:)) = bndFun;
% Initialise surface to zero
x0 = 0.3*ones(q*q, 1);
%x0 = xBar;

%==============================
% Assign function handlers
%==============================
delta = 1e-3; % for computing numerical gradient
F.f = @(x) surfaceFunc_vector(x, boundaryM, q);
F.df = @(x) surfaceGrad(x, boundaryM, q, delta);
F.d2f = @(x) surfaceHess(x, boundaryM, q, delta);

%==============================
% Newton line search
%==============================
alpha0 = 1;
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.9;
lsFun = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);
start_time = clock;
[xLS_NewtonCG, fLS_NewtonCG, nIterLS_NewtonCG, infoLS_NewtonCG] = descentLineSearch(F, 'newton-cg', lsFun, alpha0, x0, tol, maxIter);
end_time = clock;
% Measure computational time
disp(['Computational time for Newton-CG: ' num2str(etime(end_time, start_time))]);
disp(strcat('Num iter Newton-CG: ', num2str(nIterLS_NewtonCG)));

%==============================
% Interior Point - Barrier
%==============================
nCol = 7;
direction = 'x';
inequality = 'leq';
threshold = 0.1;
% Assign phi handler
phi.f = @(x) surfacePhi(x, q, nCol, direction, inequality, threshold)   + ...
             surfacePhi(x, q, nCol+1, direction, inequality, threshold) + ...
             surfacePhi(x, q, nCol-1, direction, inequality, threshold);
phi.df = @(x) surfaceGradPhi(x, q, nCol, direction, inequality, threshold) + ...
              surfaceGradPhi(x, q, nCol+1, direction, inequality, threshold) + ...
              surfaceGradPhi(x, q, nCol-1, direction, inequality, threshold);
phi.d2f = @(x) surfaceHessPhi(x, q, nCol, direction, inequality, threshold) + ...
               surfaceHessPhi(x, q, nCol+1, direction, inequality, threshold) + ...
               surfaceHessPhi(x, q, nCol-1, direction, inequality, threshold);

t = 1; %15;
mu = 10;
tol = 1e-5;
maxIter = 30;

[xBar, fBar, tBar, nIterBar, infoBar] = interiorPoint_Barrier(F, phi, x0, t, mu, tol, maxIter);

%==============================
% Interior Point - Primal Dual
%==============================
nCol = 7;
direction = 'x';
inequality = 'leq';
threshold = 0.1;
% Assign phi handler
ineqConstraint.f = @(x) surfaceIneq(x, q, nCol, direction, inequality, threshold);
ineqConstraint.df = @(x) surfaceGradIneq(x, q, nCol, direction, inequality, threshold);

mu = 10;
tol = 1e-5;
maxIter = 30;
% Initialization
lambda0 = ones(q-1, 1);
eqC = 'with-eq';
switch lower(eqC)
    case 'no-eq'
        % No equality constraint
        nu0 = zeros(0, 1);
        eqConstraint = [];
    case 'with-eq'    
        % With explicit equality constraint (fix the surface at one point) 
        index = q*6 + round(2*q/4);
        eqConstraint.A = zeros(1, q*q);
        eqConstraint.A(1, index) = 1;
        eqConstraint.b = 0.3;
        nu0 = zeros(1, 1);
end
% Backtracking options
optsBT.maxIter = 20;
optsBT.alpha = 0.1;
optsBT.beta = 0.5;

[xPD, fPD, tPD, nPD, infoPD] = interiorPoint_PrimalDual(F, ineqConstraint, eqConstraint, x0, lambda0, nu0, mu, tol, tol, maxIter, optsBT);

%==============================
% Plot residuals
%==============================
figure;
semilogy(infoPD.r_dual, 'LineWidth', 2);
hold on;
semilogy(infoPD.r_cent, 'LineWidth', 2);
%semilogy(infoPD.r_, 'LineWidth', 2);
title('Residuals');
legend('Dual residual', 'Centering residual');
grid on;

%==============================
% Plot gradient along x
%==============================
threshold = 0.1;
%load xBar;
% Newton minimisation
nCol = 7
xCol = xLS_NewtonCG(1 + (nCol-1)*q: q + (nCol-1)*q);
xGradN = (xCol(2:end) - xCol(1:end-1))*(q+1);
figure;
plot(xGradN, 'Color', 'b');
hold on;
plot(threshold + 0*xGradN, 'Color', 'r');
 
% Plot gradient for column
xCol = xBar(1 + (nCol-1)*q: q + (nCol-1)*q);
xGrad = (xCol(2:end) - xCol(1:end-1))*(q+1);
figure;
plot(xGrad, 'Color', 'b');
hold on;
% plot(xGradN, 'Color', 'g');
plot(threshold + 0*xGrad, 'Color', 'r');
title('Gradient - cubic boundary');
legend('Grad for IP', 'Grad for Newton', 'threshold');
% saveas(gcf, '../figs/Grad', 'png');
% saveas(gcf, '../figs/Grad.fig');


%==============================
% Visualize convergence
%==============================
X = 0:1/(q+1):1;
Y = 0:1/(q+1):1;

% Newton
figure
visualizeSurface(infoLS_NewtonCG, X, Y, boundaryM, 'final'); 
colorbar();
title('Newton - cubic boundary');
saveas(gcf, '../figs/SurfaceNewton', 'png');
saveas(gcf, '../figs/SurfaceNewton.fig');

% Barrier
visualizeSurface(infoBar, X, Y, boundaryM, 'final');
colorbar();
title('Interior Point - cubic boundary');
saveas(gcf, '../figs/SurfaceIP', 'png');
saveas(gcf, '../figs/SurfaceIP.fig');

% Primal Dual
figure;
visualizeSurface(infoPD, X, Y, boundaryM, 'final'); 
colorbar();
title('Primal dual - cubic boundary');


%%
%==============================
%  Subfunctions
%==============================
%%

%% descentLineSearch.m
% Wrapper function executing iteration with descent direction and line search method
%
% <include>descentLineSearch.m</include>
%%

%% surfaceFunc_vector.m
% Function to minimise the surface area. Input in Vector form
%
% <include>surfaceFunc_vector.m</include>

%% surfaceFunc_matrix.m
% Function to minimise the surface area. Input in matrix form
%
% <include>surfaceFunc_matrix.m</include>

%% surfaceGrad.m
% Gradient handler of the function to minimise the surface area
%
% <include>surfaceGrad.m</include>

%% surfaceHess.m
% Hessian handler of the function to minimise the surface area
%
% <include>surfaceHess.m</include>

%% visualizeSurface.m
% Convergence plot of iterates
%
% <include>visualizeSurface.m</include>


