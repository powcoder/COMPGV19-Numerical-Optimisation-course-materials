function visualizeConvergence(info,X,Y,Z,mode)
% VISUALIZECONVERGENCE Convergence plot of iterates
% visualizeConvergence(info,X,Y,Z,mode)
% INPUTS
% info: structure containing iteration history
%   - xs: taken steps
%   - xind: iterations at which steps were taken
%   - stopCond: shows if stopping criterium was satisfied, otherwsise k = maxIter
%   - Deltas: trust region radii
%   - rhos: relative progress
% X,Y: grid as returned by meshgrid
% Z: objective function evaluated on the grid
% mode: choose from {'final', 'iterative'}
%   'final': plot all iterates at once
%   'iterative': plot the iterates one by on to see the order in which steps are taken
% 
% Copyright (C) 2017 Marta M. Betcke, Kiko Rullan 

figure;
hold on;
% Plot contours of Z - function evaluated on grid 
contour(X, Y, Z, 20);   % just contours 
%contourf(X, Y, Z, 20); % colors inside the contours

switch mode
  case 'final'  
    % Plot all iterations
    plot(info.xs(1, :), info.xs(2, :), '-or', 'LineWidth', 2, 'MarkerSize', 3);
    title('Convergence')
    
  case 'iterative'    
    % Plot the iterates one by one to see the order in which steps are taken
    nIter = size(info.xs,2);
    
    for j = 1:nIter,            
      hold off; contour(X, Y, Z, 20); hold on      
      plot(info.xs(1, 1:j), info.xs(2, 1:j), '-or', 'LineWidth', 2, 'MarkerSize', 3);
      plot(info.xs(1, j), info.xs(2, j), '-*b', 'LineWidth', 2, 'MarkerSize', 5);
      
      if isfield(info, 'Deltas') && j > 2
        plot(info.xs(1, j-1)+cos(0:0.05:2*pi)*info.Deltas(info.xind(j)), ...
             info.xs(2, j-1)+sin(0:0.05:2*pi)*info.Deltas(info.xind(j)), ...
             ':k', 'LineWidth', 2);
      end
      
      title(['Convergence: steps 1 : ' num2str(j)])
      pause(1);
    end
         
end
