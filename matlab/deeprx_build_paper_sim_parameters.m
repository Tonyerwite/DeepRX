function simParameters = deeprx_build_paper_sim_parameters(trainNow, nFrames, carrierFrequency, modulationType)
%DEEPRX_BUILD_PAPER_SIM_PARAMETERS Paper/Table-II PUSCH settings for DeepRx.
% This wrapper keeps the project code from modifying MathWorks example files.

simParameters.TrainNow = logical(trainNow);
simParameters.UseParallel = false;
simParameters.LearnRate = 0.001;
simParameters.NTrainSamples = 30e3;
simParameters.NFrames = nFrames;

simParameters.CarrierFrequency = carrierFrequency;
simParameters.NSizeGrid = 26;
simParameters.SubcarrierSpacing = 15;
simParameters.ModulationType = string(modulationType);
simParameters.CodeRate = 658/1024;

simParameters.SNRInLimits = [-4 32];
simParameters.DelaySpreadLimits = [10e-9 300e-9];
simParameters.MaximumDopplerShiftLimits = [0 500];
simParameters.DMRSAdditionalPositionLimits = [0 1];
simParameters.DMRSConfigurationTypeLimits = [1 2];
simParameters.ChannelTrain = ["CDL-B","CDL-C","CDL-D","TDL-B","TDL-C","TDL-D"];
simParameters.ChannelValidate = ["CDL-A","CDL-E","TDL-A","TDL-E"];
end
