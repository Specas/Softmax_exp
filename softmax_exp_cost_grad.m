function [cost, grad_theta, grad_power] = softmax_exp_cost_grad(theta, p, x, y)

    m = size(x, 2);
    n = size(x, 1);
    num_classes = size(theta, 2);
    
    y_valid = zeros(num_classes, m);
    ind = sub2ind(size(y_valid), y, 1:size(y_valid, 2));
    y_valid(ind) = 1;
    
    
    prob = zeros(num_classes, m);
    for i=1:num_classes
        xt = bsxfun(@power, x, p(:, i));
        prob(i, :) = (theta(:, i)')*xt;
    end
        
    prob = bsxfun(@rdivide, prob, sum(prob));
    cost = sum(sum(log(prob).*y_valid));
    cost = -(1/m)*cost;
    
    
    grad_theta = -(1/m)*x*(y_valid - prob)';
    
    grad_power = -(1/m)*(theta.*p).*(log(x)*(y_valid - prob)');

        
    
end
    
    