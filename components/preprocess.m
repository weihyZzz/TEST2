function xR = preprocess(xR,normal_type,deavg)
switch normal_type
    case 1 % power normalization
        E=sum(xR.*conj(xR)); xR = xR./sqrt(E);
    case 2 % amplitude normalization
        normCoef = max(max(abs(xR)));
        xR = xR ./ normCoef;
end

if deavg
    xR = xR - mean(xR);
end
