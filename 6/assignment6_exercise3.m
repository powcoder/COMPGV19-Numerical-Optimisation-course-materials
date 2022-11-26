%% *COMPGV19: Assignment 6*
%
% Marta Betcke and Kiko Rul·lan
% 

%%
% *Exercise 3*
% Minimise the surface given by the boundary conditions using Algorithm 7.1

close all;
clear all;

%==============================
% Initialisation
%==============================
% Dimensions
q = 14; % dimension of interior square
n = q+2; % dimension of square with boundary
% Boundary
s = linspace(0,1-1/n,n-1);
leftBoundary   = sin(pi*s+pi);   % left boundary
rightBoundary  = sin(3*pi*s+pi); % right boundary
topBoundary    = sin(3*pi*s);    % top boundary
bottomBoundary = sin(pi*s);      % bottom boundary
bndFun = [leftBoundary topBoundary rightBoundary bottomBoundary];
figure, plot(bndFun);
% Indicies of the boundary nodes in counter clockwise order
bndInd = [1:q+2, (1:q)*(q+2) + ones(1,q)*(q+2), (q+1)*(q+2) + (q+2:-1:1), (q:-1:1)*(q+2) + ones(1,q)];
boundaryM = zeros(q+2,q+2);
boundaryM(bndInd(:)) = bndFun;
% Initialise surface to zero
x0 = zeros(q*q, 1);

%==============================
% Assign function handlers
%==============================
delta = 1e-2; % for computing numerical gradient
F.f = @(x) surfaceFunc_vector(x, boundaryM, q);   % Function handler
F.df = @(x) surfaceGrad(x, boundaryM, q, delta);  % Gradient handler
F.d2f = @(x) surfaceHess(x, boundaryM, q, delta); % Hessian handler

%==============================
% LS-Newton-CG
%==============================
% YOUR CODE HERE


%==============================
% BFGS
%==============================
% YOUR CODE HERE


%==============================
% TR-SR1
%==============================
% YOUR CODE HERE


%==============================
% Visualize convergence
%==============================
X = 0:1/(q+1):1;
Y = 0:1/(q+1):1;

% SUBSTITUTE INFO_EXAMPLE by the corresponding info structure from LS-Newton-CG, BFGS, TR-SR1 obtained in the minimisation. 
info_example.xs = zeros(q*q, 1);
visualizeSurface(info_example, X, Y, boundaryM, 'final'); 




