function J = surfaceGradIneq(x, dim, index, direction, ineq, threshold)
% SURFACEGRADINEQ computes the Jacobian of the inequalities functions for the minimal surface problem
% that bounds the value of the gradient along a direction
% J = surfaceGradIneq(x, dim, direction, ineq, threshold)
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
% grad: gradient of the inequalities function for the interior point method

% Initialize
J = zeros(dim-1, dim*dim);
% Choose direction
switch lower(direction)
    % X Direction
    case 'x'
        % Choose ineq
        switch lower(ineq)
            case 'leq'
                for i = 1:dim-1
                    a = zeros(1, dim*dim);
                    a(i + 1 + (index-1)*dim) = 1;
                    a(i + (index-1)*dim) = -1;
                    J(i, :) = a;
                end
            case 'geq'
                for i = 1:dim-1
                    a = zeros(dim*dim, 1);
                    a(i + 1 + (index-1)*dim) = 1;
                    a(i + (index-1)*dim) = -1;
                    J(i, :) = -a;
                end
        end
    % Y Direction
    case 'y'
        % Choose ineq
        switch lower(ineq)
            case 'leq'
                for i = 1:dim-1
                    a = zeros(1, dim*dim);
                    a(index + 1 + (i-1)*dim) = 1;
                    a(index + (i-1)*dim) = -1;
                    J(i, :) = a;
                end
            case 'geq'
                for i = 1:dim-1
                    a = zeros(1, dim*dim);
                    a(index + 1 + (i-1)*dim) = 1;
                    a(index + (i-1)*dim) = -1;
                    J(i, :) = -a;
                end
        end
end
 
 
