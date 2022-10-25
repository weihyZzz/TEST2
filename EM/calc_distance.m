function d = calc_distance(Param, Param_)
%{ 
This function calculates the distance between two sets of parameters. 

Input: 
    Param : old parameters
    Param_: new parameters

Output: 
    d: semi-Euclidean distance
%}

% d = norm(Param.w1 - Param_.w1) + norm(Param.w2 - Param_.w2);
d = norm(Param_.mu1-Param.mu1) + norm(Param_.mu2-Param.mu2);
 
end