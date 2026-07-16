function results = deeprx_evaluate_practical_lmmse(exampleDir, iteration, snrIn, channelModel, delaySpread, maxDopplerShift, dmrsAdditionalPosition, dmrsConfigurationType, nFrames)
%DEEPRX_EVALUATE_PRACTICAL_LMMSE Evaluate the official practical receiver.

addpath(string(exampleDir));
addpath(fileparts(mfilename("fullpath")));
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

simParameters = deeprx_build_paper_sim_parameters(false, nFrames, 4.0e9, "16QAM");
simParameters.PerfectChannelEstimator = false;
randParameters = struct();
randParameters.SNRIn = snrIn;
randParameters.ChannelModel = string(channelModel);
randParameters.DelaySpread = delaySpread;
randParameters.MaxDopplerShift = maxDopplerShift;
randParameters.DMRSAddPos = dmrsAdditionalPosition;
randParameters.DMRSConfigType = dmrsConfigurationType;

[results, ~, ~, ~] = hGetFeaturesAndLabels(simParameters, randParameters, Iteration=iteration, IndependentChannel=true);
end
