function [cost, grad] = softmax_cost_grad(theta, x, y)

    m = size(x, 2);
    n = size(x, 1);
    num_classes = size(theta, 2);
    
    y_valid = zeros(num_classes, m);
    ind = sub2ind(size(y_valid), y, 1:size(y_valid, 2));
    y_valid(ind) = 1;
    
    prob = exp(theta'*x);
    prob = bsxfun(@rdivide, prob, sum(prob));
    
    cost = sum(sum(log(prob).*y_valid));
    cost = -(1/m)*cost;
    
    grad = -(1/m)*x*(y_valid - prob)';
    
end
