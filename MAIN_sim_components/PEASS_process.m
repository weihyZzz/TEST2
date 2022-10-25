 L = min(size(s,2), size(s_est,2));
 audiowrite('peassaudio_save/estimateFile.wav',s_est(:,1:L).',fs_ref);
 audiowrite('peassaudio_save/srcFile.wav',[s(1,1:L)' s(1,1:L)'],fs_ref);
 audiowrite('peassaudio_save/interFile.wav',[s(2,1:L)' s(2,1:L)'],fs_ref);
 originalFiles={'peassaudio_save/srcFile.wav';'peassaudio_save/interFile.wav'};%仿真中是默认第一路为target,其他路为inter
 estimateFile = 'peassaudio_save/estimateFile.wav';
 options.destDir = 'peassaudio_save/';
 options.segmentationFactor = 1; % increase this integer if you experienced "out of memory" problems
 res = PEASS_ObjectiveMeasure(originalFiles,estimateFile,options);
%%%%%%%%%%%%%%%%%
% Display results
%%%%%%%%%%%%%%%%%

fprintf('************************\n');
fprintf('* INTERMEDIATE RESULTS *\n');
fprintf('************************\n');

fprintf('The decomposition has been generated and stored in:\n');
cellfun(@(s)fprintf(' - %s\n',s),res.decompositionFilenames);

fprintf('The ISR, SIR, SAR and SDR criteria computed with the new decomposition are:\n');
fprintf(' - SDR = %.1f dB\n - ISR = %.1f dB\n - SIR = %.1f dB\n - SAR = %.1f dB\n',...
    res.SDR,res.ISR,res.SIR,res.SAR);

fprintf('The audio quality (PEMO-Q) criteria computed with the new decomposition are:\n');
fprintf(' - qGlobal = %.3f\n - qTarget = %.3f\n - qInterf = %.3f\n - qArtif = %.3f\n',...
    res.qGlobal,res.qTarget,res.qInterf,res.qArtif);

fprintf('*************************\n');
fprintf('****  FINAL RESULTS  ****\n');
fprintf('*************************\n');
fprintf(' - Overall Perceptual Score: OPS = %.f/100\n',res.OPS)
fprintf(' - Target-related Perceptual Score: TPS = %.f/100\n',res.TPS)
fprintf(' - Interference-related Perceptual Score: IPS = %.f/100\n',res.IPS)
fprintf(' - Artifact-related Perceptual Score: APS = %.f/100\n',res.APS);
