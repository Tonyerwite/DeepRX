function results = deeprx_evaluate_known_channel_lmmse(exampleDir, iteration, snrIn, channelModel, delaySpread, maxDopplerShift, dmrsAdditionalPosition, dmrsConfigurationType, nFrames)
%DEEPRX_EVALUATE_KNOWN_CHANNEL_LMMSE Evaluate the official known-channel LMMSE receiver.

addpath(string(exampleDir));
addpath(fileparts(mfilename("fullpath")));
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

simParameters = deeprx_build_paper_sim_parameters(false, nFrames, 4.0e9, "16QAM");
simParameters.PerfectChannelEstimator = true;
randParameters = struct();
randParameters.SNRIn = snrIn;
randParameters.ChannelModel = string(channelModel);
randParameters.DelaySpread = delaySpread;
randParameters.MaxDopplerShift = maxDopplerShift;
randParameters.DMRSAddPos = dmrsAdditionalPosition;
randParameters.DMRSConfigType = dmrsConfigurationType;

[results, ~, ~, ~] = hGetFeaturesAndLabels(simParameters, randParameters, Iteration=iteration);
end
