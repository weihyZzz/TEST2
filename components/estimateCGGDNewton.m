function [c,besterr] = estimateCGGDNewton(Xa)
%% MLE
 bestC = max(.5,.1*randn+1); %start at Gaussian
%if moment == 1
    %Moment estimator for c to get in ball park
    bestVal = 1e10;%��С���
    x = real(Xa(1,:));y = imag(Xa(1,:));%ȡʵ���鲿
    mm = mean(x.^4)/mean(x.^2)^2 + mean(y.^4)/mean(y.^2)^2;%��Ӧ��ʽ9ǰ��k(c)
    for cc =[0.1:0.001:4] % ML����֮ǰ����c�Ĺ���ֵ������0.1��4֮��  [[.1:.1:.45] [.5:.5:1.5] [2:4]]
        temp = 3*gamma(1/cc)*gamma(3/cc)/gamma(2/cc)^2 -mm; %��Ӧ��ʽ9���f(c)
        if abs(temp) < bestVal
            bestVal = abs(temp);
            bestC = cc;%��ʱ��Ӧ��cc��Ϊ���ֵ
        end
    end
c = bestC;besterr=bestVal ;
end