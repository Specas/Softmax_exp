function accuracy = find_accuracy(theta, x,y) 
  [~,labels] = max(theta'*x, [], 1);

  correct=sum(y == labels);
  accuracy = correct / length(y);
