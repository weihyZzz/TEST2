function p = prob(point, beta)
%{
%% calculate probability / likelihood
% Probability that given a set of parameters \theta for the PDFs the data X can be
% observed.; this is equivalent of the likelihood of the parameters \theta
% given data points X
%
% point = [x y]
% mu = 1x2
% sigma = 2x2
% lambda = 1x2
% sigma has to be square (actually semi definite positive)
% TODO: check sigma for that and fix it if it is not.
%}
F = length(point);
p = -norm(point)^beta - log_gamma(2*F/beta+1) + log_gamma(F+1) - F*log(pi);
end

function lg = log_gamma(z)
lg = (z-1/2)*log(z) - z + 1/2*log(2*pi) + 1/12*z - 1/(360*z^3) + 1/(1260*z^5);
end