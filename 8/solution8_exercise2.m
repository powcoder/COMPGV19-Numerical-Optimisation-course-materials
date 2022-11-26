%% *COMPGV19: Tutorial 7*
%
% Marta Betcke and Kiko Rul·lan
% 
close all;
clear all;

%%
% *Exercise 2*
% Consider the problem of minimising the function 
% q(x, y) = (x - 2y)^2 + (x - 2)^2
% subject to 
% x - y - 4 <= 0

%==============================
% Function and boundary conditions
%==============================
% Minimisation function
F.f = @(x) (x(1) - 2*x(2))^2 + (x(1)-2)^2;
F.df = @(x) [2*(x(1) - 2*x(2)) + 2*(x(1) - 2); -4*(x(1) - 2*x(2))];
F.d2f = @(x) [4, -4; -4, 8];

% Choose boundary
boundary = 'inequality';
slope = 1;
orden = 3;
switch lower(boundary)
    case 'inequality'
        % Inequality constraint
        ineqConstraint.f = @(x) slope*x(1) - x(2) + orden;
        ineqConstraint.df = @(x) [slope, -1];
        % No equality constraint
        eqConstraint = [];
        % Initialization
        x0 = [-4; 5];
        lambda0 = 1;
        nu0 = zeros(0, 1);
    case 'equality'
        % Equality constraint
        %ineqConstraint.f = [];
        ineqConstraint.f = @(x) slope*x(1) - x(2) + orden-1;
        ineqConstraint.df = @(x) [slope, -1];
        % No equality constraint
        eqConstraint.A = [slope, -1];
        eqConstraint.b = -orden;
        % Initialization
        x0 = [-4; -4*(slope)+orden];
        lambda0 = 1;
        nu0 = 1;
end
%==============================
% Parameters and minimisation
%==============================
% Set parameters
mu = 10;
tol = 1e-3;
maxIter = 30;
% Backtracking options
optsBT.maxIter = 20;
optsBT.alpha = 0.1;
optsBT.beta = 0.5;
% Run the iterations
[xPD, fPD, tPD, nPD, infoPD] = interiorPoint_PrimalDual(F, ineqConstraint, eqConstraint, x0, lambda0, nu0, mu, tol, tol, maxIter, optsBT);
disp(['xPD = [' num2str(xPD(1)) ' ' num2str(xPD(2)) ']']);
% Visualise the results
x = [-8:0.1:8];
y = [-8:0.1:8]';
[X, Y] = meshgrid(x, y);
q = @(x, y) (x - 2*y).^2 + (x - 2).^2;
Z = log(q(X, Y)+1);
visualizeConvergence(infoPD, X, Y, Z, 'final');
hold on;
plot(x, slope*x + orden, 'Color', 'g', 'LineWidth', 1);
%if (strcmp(boundary, 'equality'))
%    plot(x, slope*x + orden-1, 'Color', 'm', 'LineWidth', 1);
%end
axis([x(1) x(end) y(1) y(end)]); 

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
