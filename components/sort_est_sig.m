function label = sort_est_sig(s_est)
[spec_coeff_num, frame_num, source_num] = size(s_est);
detect_range = round(0*spec_coeff_num)+1:round(0.2*spec_coeff_num);
s1_energy = sum(sum(abs(s_est(detect_range,:,1)).^2));
s2_energy = sum(sum(abs(s_est(detect_range,:,2)).^2));
if s1_energy >= s2_energy
    label{1} = 'interference'; label{2} = 'target';
else
    label{1} = 'target'; label{2} = 'interference';
end