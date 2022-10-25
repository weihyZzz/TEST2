function Data = expectation(Data, Param)
%{ 
This function calculates the first step of the EM algorithm, Expectation.
It calculates the probability of each specific data point belong to each
cluster or class

Input: 
    Data : nx3 (number of data points , [x, y, label])
    Param: (mu, sigma, lambda)

Output: 
    Data: the dataset with updated labels
%}

for ii = 1: size(Data,1)
    x = Data(ii, 1:end-1);
    
    p_cluster1 = prob(x, Param.n_orders(1));
    p_cluster2 = prob(x, Param.n_orders(2));
    
    if p_cluster1 > p_cluster2
        Data(ii, end) = 1;
    else
        Data(ii, end) = 2;
    end
end
end