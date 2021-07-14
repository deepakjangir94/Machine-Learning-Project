function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples % 12

% You need to return the following variables correctly 
J = 0; % 1 scaler
grad = zeros(size(theta)); % (2x1)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% X = (12x2), y = (12x1), theta = (2x1), lambda = scaler

J = (1/(2*m))*sum((X*theta - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2); % sum((12x2)*(2x1) - (12x1)) + sum(2x1) = sum((12x1) - (12x1)) + sum(2x1) = sum(12x1) + sum(2x1) = 1 scaler

grad(1) = (1/m)*(X(:, 1)'*(X*theta - y)) ; % (12x1)'*((12x2)*(2x1) - (12x1)) = (1x12)((12x1) - (12x1)) = (1x12)(12x1) = (1x1) = scaler
grad(2:end) = (1/m)*(X(:, 2:end)'*(X*theta - y)) + (lambda/m)*theta(2:end); % (12x1)'*((12x2)*(2x1) - (12x1)) = (1x12)((12x1) - (12x1)) = (1x12)(12x1) = (1x1) = scaler









% =========================================================================

grad = grad(:);

end
