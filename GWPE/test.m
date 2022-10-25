multi_channel_wav=[wav_dir wav_name];
[multi_wav, fs] = audioread(multi_channel_wav);

% fft config
fft_config.frame_len = 512;
fft_config.frame_shift = 128;
fft_config.fft_len = fft_config.frame_len ;

% GWPE config
gwpe_config.K = 30;
gwpe_config.delta=2;
gwpe_config.iterations = 50; 

% GWPE dereverb
[original_spec, dereverb_spec,dereverb_wav,  weight ] = GWPE( multi_wav, gwpe_config, fft_config);
audiowrite([wav_dir 'der_' wav_name], dereverb_wav, fs);
