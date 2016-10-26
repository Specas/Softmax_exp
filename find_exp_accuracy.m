function accuracy = find_exp_accuracy(theta, p, x, y) 

    m = size(x, 2);
    n = size(x, 1);
    num_classes = size(theta, 2);
  
    prob = zeros(num_classes, m);
    for i=1:num_classes
        xt = bsxfun(@power, x, p(:, i));
        prob(i, :) = (theta(:, i)')*xt;
    end
    
    [~,labels] = max(prob, [], 1);

    correct=sum(y == labels);
    accuracy = correct / length(y);
    
end
