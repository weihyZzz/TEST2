function [c,besterr] = estimateCGGDNewton(Xa)
%% MLE
 bestC = max(.5,.1*randn+1); %start at Gaussian
%if moment == 1
    %Moment estimator for c to get in ball park
    bestVal = 1e10;%最小误差
    x = real(Xa(1,:));y = imag(Xa(1,:));%取实部虚部
    mm = mean(x.^4)/mean(x.^2)^2 + mean(y.^4)/mean(y.^2)^2;%对应公式9前的k(c)
    for cc =[0.1:0.001:4] % ML估计之前给出c的估计值，遍历0.1到4之间  [[.1:.1:.45] [.5:.5:1.5] [2:4]]
        temp = 3*gamma(1/cc)*gamma(3/cc)/gamma(2/cc)^2 -mm; %对应公式9后的f(c)
        if abs(temp) < bestVal
            bestVal = abs(temp);
            bestC = cc;%此时对应的cc极为最佳值
        end
    end
c = bestC;besterr=bestVal ;
end