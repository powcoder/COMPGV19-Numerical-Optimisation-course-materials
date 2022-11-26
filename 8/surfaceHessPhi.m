function hess = surfaceHessPhi(x, dim, index, direction, ineq, threshold)
% SURFACEHESSPHI computes the Hessian of the phi function for the minimal surface problem
% that bounds the value of the gradient along a direction
% val = surfaceGradPhi(x, dim, direction, ineq, threshold)
% 
% INPUTS
% x: vector to evaluate the surface
% dim: dimension
% index: index to fix the inequality
% direction: choose between {'x', 'y'} for fixing the axis of the constraint
% ineq: choose between {'leq' (Ax <= b), 'geq' (Ax >= b)}
% threshold: value for b
% 
% OUTPUTS
% hess: Hessian of the phi function for the interior point method

% Initialize
hess = zeros(dim*dim);
% Choose direction
switch lower(direction)
    % X Direction
    case 'x'
        % Choose ineq
        switch lower(ineq)
            case 'leq'
                for i = 1:dim-1
                    a = zeros(dim*dim, 1);
                    a(i + 1 + (index-1)*dim) = 1;
                    a(i + (index-1)*dim) = -1;
                    xDif = (x(i + 1 + (index-1)*dim) - x(i + (index-1)*dim))*(dim+1);
                    hess = hess + 1/((threshold - xDif)^2)*a*(a');
                end
            case 'geq'
                for i = 1:dim-1
                    a = zeros(dim*dim, 1);
                    a(i + 1 + (index-1)*dim) = 1;
                    a(i + (index-1)*dim) = -1;
                    xDif = (x(i + 1 + (index-1)*dim) - x(i + (index-1)*dim))*(dim+1);
                    hess = hess + 1/((xDif - threshold)^2)*a*(a');
                end
        end
    % Y Direction
    case 'y'
        % Choose ineq
        switch lower(ineq)
            case 'leq'
                for i = 1:dim-1
                    a = zeros(dim*dim, 1);
                    a(index + 1 + (i-1)*dim) = 1;
                    a(index + (i-1)*dim) = -1;
                    xDif = (x(index + 1 + (i-1)*dim) - x(index + (i-1)*dim))*(dim + 1);
                    hess = hess + 1/((threshold - xDif)^2)*a*(a');
                end
            case 'geq'
                for i = 1:dim-1
                    a = zeros(dim*dim, 1);
                    a(index + 1 + (i-1)*dim) = 1;
                    a(index + (i-1)*dim) = -1;
                    xDif = (x(index + 1 + (i-1)*dim) - x(index + (i-1)*dim))*(dim + 1);
                    hess = hess + 1/((threshold - xDif)^2)*a*(a');
                end
        end
end
 
 
