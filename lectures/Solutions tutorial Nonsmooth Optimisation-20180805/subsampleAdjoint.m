function y = subsampleAndjoint(x,N,ind) 
y = zeros(N,1);
y(ind) = x;
end
