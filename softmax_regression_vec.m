function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);


  % theta is a vector;  need to reshape to n x num_classes-1. 

  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%% 

y_c = zeros(num_classes, m);
ind = sub2ind(size(y_c), y, 1:size(y_c, 2));
y_c(ind) = 1; %Only for the classes present in that set

prob = exp(theta'*X);
prob = [prob; ones(1, m)]; %As the last class has theta = 0
prob = bsxfun(@rdivide, prob, sum(prob));

f = sum(sum(log(prob).*y_c));
f = -f;

g = -X*(y_c - prob)';
g(:, end) = [];

  
  g=g(:); % make gradient a vector for minFunc

