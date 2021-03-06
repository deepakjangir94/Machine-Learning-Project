function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
reg_theta = theta.^2;
reg_theta(1) = 0;
J = (1/m)*(-y'*log(sigmoid(X*theta)) - (ones(1, m) - y')*log(ones(m, 1) - sigmoid(X*theta))) + (lambda/(2*m))*sum(reg_theta);

reg_theta = theta.^2;
reg_theta(1) = 0;
J = (1/m)*(-y'*log(sigmoid(X*theta)) - (ones(m, 1) - y)'*log(ones(m, 1) - sigmoid(X*theta))) + (lambda/(2*m))*sum(reg_theta);


pseudo_grad2 = (1/m)*(X'*(sigmoid(X*theta) - y)) + (lambda/m)*theta;
pseudo_grad1 = ((1/m)*(X'*(sigmoid(X*theta) - y)));
pseudo_grad2(1) = pseudo_grad1(1);
grad = pseudo_grad2;
% =============================================================

end
