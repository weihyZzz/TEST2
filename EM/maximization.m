function Param = maximization(Data, Param)
%{ 
This function calculates the second step of the EM algorithm, Maximization.
It updates the parameters of the Normal distributions according to the new 
labled dataset.

Input: 
    Data : nx3 (number of data points , [x, y, label])
    Param: (mu, sigma, lambda)

Output: 
    Param: updated parameters 
%}

points_in_cluster1 = Data(Data(:,end) == 1,:);
points_in_cluster2 = Data(Data(:,end) == 2,:);

Param.w1 = size(points_in_cluster1,1) / size(Data,1);
Param.w2 = size(points_in_cluster2,1) / size(Data,1);

Param.mu1 = mean(points_in_cluster1(:,1));
Param.mu2 = mean(points_in_cluster2(:,1));

% Param.n_orders(1) = estimateCGGDNewton(points_in_cluster1(:,1)');
% Param.n_orders(2) = estimateCGGDNewton(points_in_cluster2(:,1)');

end