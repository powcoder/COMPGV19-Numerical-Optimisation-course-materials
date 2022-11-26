function grad = surfaceGradPhi(x, dim, index, direction, ineq, threshold)
% SURFACEGRADPHI computes the gradient of the phi function for the minimal surface problem
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
% grad: gradient of the phi function for the interior point method

% Initialize
grad = zeros(dim*dim, 1);
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
                    grad = grad + 1/(threshold - xDif)*a;
                end
            case 'geq'
                for i = 1:dim-1
                    a = zeros(dim*dim, 1);
                    a(i + 1 + (index-1)*dim) = 1;
                    a(i + (index-1)*dim) = -1;
                    xDif = (x(i + 1 + (index-1)*dim) - x(i + (index-1)*dim))*(dim+1);
                    grad = grad + 1/(xDif - threshold)*a;
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
                    grad = grad + 1/(threshold - xDif)*a;
                end
            case 'geq'
                for i = 1:dim-1
                    a = zeros(dim*dim, 1);
                    a(index + 1 + (i-1)*dim) = 1;
                    a(index + (i-1)*dim) = -1;
                    xDif = (x(index + 1 + (i-1)*dim) - x(index + (i-1)*dim))*(dim + 1);
                    grad = grad + 1/(xDif - threshold)*a;
                end
        end
end
 
 
