clc;
clear all;

addpath common

binary_digits = false;
[train,test] = ex1_load_mnist(binary_digits);

train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.

num_classes = max(max(train.y));

m=size(train.X,2);
n=size(train.X,1);

theta = rand(n,num_classes);
alpha = 2;
momentum = 0;




% cost1 = softmax_cost(theta, train.X, train.y);
% grad = softmax_gradient(theta, train.X, train.y);
% 
% disp(cost1);
% % disp(grad);
% 
% theta = theta - alpha*grad;
% 
% cost2 = softmax_cost(theta, train.X, train.y);
% grad = softmax_gradient(theta, train.X, train.y);
% 
% disp(cost2);
% % disp(grad);



accuracy = find_accuracy(theta, train.X, train.y);
disp(accuracy);




for i=1:200
    
    [cost, grad] = softmax_cost_grad(theta, train.X, train.y);
    
    fprintf('Cost: %f Iteration: %d\n', cost, i);
    
    theta = theta - (theta*momentum + alpha*grad);
    
end

accuracy = find_accuracy(theta, train.X, train.y);
fprintf('Accuracy: %f\n', accuracy*100);


