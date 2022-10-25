function w = calc_mix_prob(y,beta,w)
F = 1;
z1 = -norm(y)^beta(1) - log_gamma(2*F/beta(1)+1) + log_gamma(F+1) - F*log(pi) - log(w(1));
z2 = -norm(y)^beta(2) - log_gamma(2*F/beta(2)+1) + log_gamma(F+1) - F*log(pi) - log(w(2));
log_sum_exp = max(z1,z2) + log(exp(z1-max(z1,z2))+exp(z2-max(z1,z2)));

log_w1 = z1 - log_sum_exp; w(1) = exp(log_w1);
log_w2 = z2 - log_sum_exp; w(2) = exp(log_w2);
end

function lg = log_gamma(z)
lg = (z-1/2)*log(z) - z + 1/2*log(2*pi) + 1/12*z - 1/(360*z^3) + 1/(1260*z^5);
end