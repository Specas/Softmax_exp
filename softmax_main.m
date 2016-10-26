clc;
clear all;

addpath common

binary_digits = false;
[train,test] = ex1_load_mnist(binary_digits);

train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.

xtrain = train.X;
ytrain = train.y;

xtest = test.X;
ytest = test.y;





maxx = max(max(xtrain));
minx = min(min(xtrain));
meanx = mean2(xtrain);
stdx = std2(xtrain);

a = 0.0001;
b = 1;

xtrain = a + (xtrain-minx)*(b-a)/(maxx - minx);
% xtrain = (xtrain-minx)/stdx;

% disp(min(min(xtrain)));
% disp(max(max(xtrain)));


num_classes = max(max(train.y));

m=size(train.X,2);
n=size(train.X,1);

mini_batch_size = 100;

theta = rand(n, num_classes);
p = ones(n, num_classes);

alpha_theta = 20;
alpha_power = 0.005;
momentum_theta = 0.0;
momentum_power = 0.0;

d = [xtrain', ytrain'];

clear xtrain;
clear ytrain;
clear train;




for i=1:500
    
    
%     d = d(randperm(size(d, 1)), :);
    sam = datasample(1:m, mini_batch_size, 'Replace', false);
    
    xtrain_mini = d(sam, 1:end-1)';
    ytrain_mini = d(sam, end)';
    
%     xtrain_mini = d(1:mini_batch_size, 1:end-1)';
%     ytrain_mini = d(1:mini_batch_size, end)';
     
    
%     [cost, grad] = softmax_cost_grad(theta, xtrain_mini, ytrain_mini);
%     
%     fprintf('Cost: %f Iteration: %d\n', cost, i);
%     
%     theta = theta - (theta*momentum + alpha*grad);


    [cost, grad_theta, grad_power] = softmax_exp_cost_grad(theta, p, xtrain_mini, ytrain_mini);
    
    fprintf('Cost: %f Iteration: %d\n', cost, i);
    
    theta = theta - (theta*momentum_theta + alpha_theta*grad_theta);
    p = p - (p*momentum_power + alpha_power*grad_power);
    
end

accuracy = find_exp_accuracy(theta, p,  d(:, 1:end-1)', d(:, end)');
fprintf('Accuracy: %f\n', accuracy*100);


