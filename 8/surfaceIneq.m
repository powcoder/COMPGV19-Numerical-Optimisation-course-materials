function val = surfaceIneq(x, dim, index, direction, ineq, threshold)
% SURFACEINEQ computes the inequality function for the minimal surface problem
% that bounds the value of the gradient along a direction
% val = surfaceIneq(x, dim, direction, ineq, threshold)
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
% val: evaluation of the inequalities function for the interior point method

% Initialize
val = zeros(dim-1, 1);
% Choose direction
switch lower(direction)
    % X Direction
    case 'x'
        % Choose ineq
        switch lower(ineq)
            case 'leq'
                for i = 1:dim-1
                    grad = (x(i + 1 + (index-1)*dim) - x(i + (index-1)*dim))*(dim+1);
                    val(i) = grad - threshold;
                end
            case 'geq'
                for i = 1:dim-1
                    grad = (x(i + 1 + (index-1)*dim) - x(i + (index-1)*dim))*(dim+1);
                    val(i) = threshold - grad;
                end
        end
    % Y Direction
    case 'y'
        % Choose ineq
        switch lower(ineq)
            case 'leq'
                for i = 1:dim-1
                    grad = (x(index + 1 + (i-1)*dim) - x(index + (i-1)*dim))*(dim + 1);
                    val(i) = grad - threshold;
                end
            case 'geq'
                for i = 1:dim-1
                    grad = (x(index + 1 + (i-1)*dim) - x(index + (i-1)*dim))*(dim + 1);
                    val(i) = threshold - grad;
                end
        end
end
 
 
