function [xMin, fMin, t, nIter, infoPD] = interiorPoint_PrimalDual(F, ineqConstraint, eqConstraint, x0, lambda0, nu0, mu, tol, tolFeas, maxIter, opts)
% INTERIORPOINT_BARRIER function to minimise a quadratic form with constraints
% [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%  - f: function to minimise
%  - df: gradient of function 
%  - d2f: Hessian of function 
% ineqConstraint: structure with fields
%  - f: vectorial linear function that sets the inequality constraints
%  - df: Jacobian of the vectorial function f
% eqConstraint: structure with fields
%  - A: matrix for the equality constraints
%  - b: vector for the equality constraints
% x0: initial iterate
% lambda0: initial lambda vector
% nu0: initial nu
% mu: increase factor for t
% tol: stopping condition on the value of m/t
% tolFeas: feasibility tolerance
% maxIter: maximum number of iterations
% opts: options for backtracking
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% infoPD: structure with information about the iteration 
%   - xs: iterate history for x 
%   - ys: iterate history for y
%
% Copyright (C) 2017  Kiko Rul·lan, Marta M. Betcke

% Dimensions
n = length(x0);
% Check if the inequality constraints are set
if (~isfield(ineqConstraint, 'f'))
    ineqConstraint.f = @(x) -inf;
end
if (~isfield(ineqConstraint, 'df'))
    ineqConstraint.df = @(x) zeros(1, n);
end
m = size(lambda0, 1);
% Check if the equality constraints are set
if (~isfield(eqConstraint, 'A'))
    eqConstraint.A = zeros(0, n);
end
if (~isfield(eqConstraint, 'b'))
    eqConstraint.b = zeros(0, 1);
end
nEq = size(eqConstraint.A, 1);

% Initilize
nIter = 0;
stopCond = false;
x_k = x0;
l_k = lambda0;
n_k = nu0;
infoPD.xs = x_k;
infoPD.r_dual = [];
infoPD.r_cent = [];
infoPD.r_prim = [];

% Define residual function
r_dual = @(t, x, l, n) F.df(x) + (ineqConstraint.df(x))'*l + eqConstraint.A'*n;
r_cent = @(t, x, l, n) -diag(l)*ineqConstraint.f(x) - ones(m, 1)/t;
r_prim = @(t, x, l, n) eqConstraint.A*x - eqConstraint.b;
res = @(t, x, l, n) [r_dual(t, x, l, n); r_cent(t, x, l, n); r_prim(t, x, l, n)];

% Surrogate gap
eta = - (ineqConstraint.f(x0))'*lambda0;
t = mu*m/eta;
% Loop 
while (~stopCond && nIter < maxIter)
    disp(['Iteration ' int2str(nIter)]);

    % Compute residual
    res_k = res(t, x_k, l_k, n_k);
    % Find direction
    deltaY = -[F.d2f(x_k)                       ,   ineqConstraint.df(x_k)'    , eqConstraint.A'; ...
               -diag(l_k)*ineqConstraint.df(x_k),  -diag(ineqConstraint.f(x_k)), zeros(m, nEq)  ; ...
               eqConstraint.A                   ,   zeros(nEq, m)              , zeros(nEq, nEq)]\res_k;
             
    % Line search - Backtracking
    deltaX = deltaY(1:n);
    deltaL = deltaY(n+1:n+m); 
    deltaN = deltaY(n+m+1:end); 
    l_k_neg = l_k(deltaL < 0)./deltaL(deltaL < 0);
    if (length(l_k_neg) > 0)
        s_max = min(1, min(-l_k_neg));
    else 
        s_max = 1;
    end
    s = 0.99*s_max;
    nIterBT = 0;
    stopCondBT = false;
    while(~stopCondBT && nIterBT < opts.maxIter)
        if ((max(ineqConstraint.f(x_k + s*deltaX)) < 0) && ... 
            (norm(res(t, x_k + s*deltaX, l_k + s*deltaL, n_k + s*deltaN)) <= (1 - opts.alpha*s)*norm(res_k)))
            stopCondBT = true;
        end       
        s = opts.beta*s;
        nIterBT = nIterBT + 1;
    end

    % Update point
    x_k = x_k + s*deltaX;
    l_k = l_k + s*deltaL;
    n_k = n_k + s*deltaN;

    % Surrogate gap
    eta = - (ineqConstraint.f(x_k)')*l_k;
    t = mu*m/eta;
    % Check stopping condition
    if ((eta < tol) && (norm(r_dual(t, x_k, l_k, n_k)) <= tolFeas) && (norm(r_prim(t, x_k, l_k, n_k)) <= tolFeas))
        stopCond = true; 
    end;

    % Store info
    infoPD.xs = [infoPD.xs x_k];
    infoPD.r_dual = [infoPD.r_dual norm(r_dual(t, x_k, l_k, n_k))];
    infoPD.r_cent = [infoPD.r_cent norm(r_cent(t, x_k, l_k, n_k))];
    infoPD.r_prim = [infoPD.r_prim norm(r_prim(t, x_k, l_k, n_k))];
    % Increment number of iterations
    nIter = nIter + 1;
end

% Assign values
xMin = x_k;
fMin = F.f(x_k);

