function [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter)
% DESCENTLINESEARCH Wrapper function executing  descent with line search
% [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% descent: specifies descent direction {'steepest', 'newton', 'newton-cg', 'bfgs'}
% ls: specifies the line search to apply
% alpha0: initial step length 
% x0: initial iterate
% tol: stopping condition on minimal allowed step
%      norm(x_k - x_k_1)/norm(x_k) < tol;
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% info: structure with information about the iteration 
%   - xs: iterate history 
%   - alphas: step lengths history 
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rullan

% Parameters
% Stopping condition {'step', 'grad'}
stopType = 'step';

% Extract inverse Hessian approximation handler
extractH = 1;

% Initialization
nIter = 0;
x_k = x0;
info.xs = x0;
info.alphas = alpha0;
stopCond = false; 

switch lower(descent)
  case 'bfgs'
    H_k = @(y) y;
    % Store H matrix in columns
    info.H = [];
end

% Loop until convergence or maximum number of iterations
while (~stopCond && nIter <= maxIter)
    
  % Increment iterations
    nIter = nIter + 1;
    % Compute descent direction
    switch lower(descent)
      case 'steepest'
        p_k = -F.df(x_k); % steepest descent direction
      case 'newton'
        p_k = -F.d2f(x_k)\F.df(x_k); % Newton direction
        if p_k'*F.df(x_k) > 0 % force to be descent direction (only active if F.d2f(x_k) not pos.def.)
          p_k = -p_k;
        end
      case 'newton-cg'
        % Conjugate gradient method
        df_k = F.df(x_k); % gradient
        B_k = F.d2f(x_k); % hessian
        eps_k = min(0.5, sqrt(norm(df_k)))*norm(df_k);
        z_j = 0*df_k;
        r_j = df_k;
        d_j = -df_k;
        stopCondCG = false;
        maxIterCG = 200; % maxIter
        nIterCG = 0;
        while (~stopCondCG & nIterCG <= maxIterCG)
          if (d_j)'*B_k*d_j <= 0
            if nIterCG == 0; p_k = d_j;
            else p_k = z_j; end;
            stopCondCG = true;
            eps_k = 0;
          end
          norm_r_j = r_j'*r_j;
          a_j = norm_r_j/(d_j'*B_k*d_j);
          z_j = z_j + a_j*d_j;
          r_j = r_j + a_j*B_k*d_j;
          if sqrt(r_j'*r_j) < eps_k || nIterCG == maxIterCG; 
            stopCondCG = true; 
            p_k = z_j;
          end;
          b_j = r_j'*r_j/norm_r_j;
          d_j = -r_j + b_j*d_j;
          nIterCG = nIterCG + 1;
        end
      case 'bfgs'
        p_k = -H_k(F.df(x_k));
    end
    
    % Call line search given by handle ls for computing step length
    alpha_k = ls(x_k, p_k, alpha0);
    
    % Update x_k and f_k
    x_k_1 = x_k;
    x_k = x_k + alpha_k*p_k;
    
    switch lower(descent)
      case 'bfgs'
        % Compute s_k and y_k
        s_k = x_k - x_k_1;
        y_k = F.df(x_k) - F.df(x_k_1);
        % Verify s_k^T*y_k > 0
        if s_k'*y_k <= 0
          error('Positivity condition: <s_k,y_k> > 0 not satisfied. Check if LS satisfied Wolfe conditions.');
        end
        
        if nIter == 1
          % Update initial guess H_0. Note that initial p_0 = -F.df(x_0) and x_1 = x_0 + alpha * p_0.
          disp(['Rescaling H0 with ' num2str((s_k'*y_k)/(y_k'*y_k)) ])
          H_k = @(x) (s_k'*y_k)/(y_k'*y_k) * x;
        end
        
        % Efficient update H
        rho_k = 1/(s_k'*y_k);
        projL = @(x) (x - rho_k*s_k*(y_k'*x));
        projR = @(x) (x - rho_k*y_k*(s_k'*x));
        H_k = @(x) projL( H_k( projR(x) ) ) + rho_k*s_k*(s_k'*x);

        if extractH
            % Extraction of H_k as handler
            info.H{length(info.H)+1} = H_k;
        end
    end
    

    % Store iteration info
    info.xs = [info.xs x_k];
    info.alphas = [info.alphas alpha_k];
    
    switch stopType
      case 'step' 
        % Compute relative step length
        normStep = norm(x_k - x_k_1)/norm(x_k_1);
        stopCond = (normStep < tol);
      case 'grad'
        stopCond = (norm(F.df(x_k), 'inf') < tol*(1 + abs(F.f(x_k))));
    end
    
end

% Assign output values 
xMin = x_k;
fMin = F.f(x_k); 

