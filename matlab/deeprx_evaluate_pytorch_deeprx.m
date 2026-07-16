function results = deeprx_evaluate_pytorch_deeprx(pytorchExampleDir, pythonPath, modelPath, iteration, snrIn, channelModel, delaySpread, maxDopplerShift, dmrsAdditionalPosition, dmrsConfigurationType, nFrames)
%DEEPRX_EVALUATE_PYTORCH_DEEPRX Evaluate a PyTorch DeepRx checkpoint.

addpath(string(pytorchExampleDir));
addpath(fileparts(mfilename("fullpath")));
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

oldFolder = pwd;
cleanup = onCleanup(@() cd(oldFolder));
cd(string(pytorchExampleDir));

simParameters = deeprx_build_paper_sim_parameters(false, nFrames, 4.0e9, "16QAM");
simParameters.PerfectChannelEstimator = false;
simParameters.PythonExecutionMode = "OutOfProcess";
simParameters.PythonPath = string(pythonPath);
simParameters.PythonRequirements = fullfile(fileparts(mfilename("fullpath")), "requirements_6GVerify_project.txt");

simParameters = hGetAdditionalSystemParameters(simParameters, DisplayParameterSummary=false);
simParameters.ModelPath = string(modelPath);
simParameters.ModelInput = simParameters.InputSize;
simParameters.ModelOutput = [simParameters.ModelInput(1:2), 4];

net = hCreateTorchDeepRx(simParameters, Evaluate=true);

randParameters = struct();
randParameters.SNRIn = snrIn;
randParameters.ChannelModel = string(channelModel);
randParameters.DelaySpread = delaySpread;
randParameters.MaxDopplerShift = maxDopplerShift;
randParameters.DMRSAddPos = dmrsAdditionalPosition;
randParameters.DMRSConfigType = dmrsConfigurationType;

[results, ~, ~, ~] = hGetFeaturesAndLabels(simParameters, randParameters, Net=net, Iteration=iteration, IndependentChannel=true);
end
