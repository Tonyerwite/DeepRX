function batch = deeprx_export_official_batch(exampleDir, iteration, snrIn, channelModel, delaySpread, maxDopplerShift, dmrsAdditionalPosition, dmrsConfigurationType, nFrames)
%DEEPRX_EXPORT_OFFICIAL_BATCH Generate one official MATLAB DeepRx batch.
% The output arrays use MATLAB layout:
%   X          [F S C N]
%   TargetBits [F S B N]
%   DataMask   [F S 1 N]

addpath(string(exampleDir));
addpath(fileparts(mfilename("fullpath")));
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

simParameters = deeprx_build_paper_sim_parameters(true, nFrames, 4.0e9, "16QAM");
randParameters = struct();
randParameters.SNRIn = snrIn;
randParameters.ChannelModel = string(channelModel);
randParameters.DelaySpread = delaySpread;
randParameters.MaxDopplerShift = maxDopplerShift;
randParameters.DMRSAddPos = dmrsAdditionalPosition;
randParameters.DMRSConfigType = dmrsConfigurationType;

[~, X, T, simParameters] = hGetFeaturesAndLabels(simParameters, randParameters, Iteration=iteration);
X = local_extract_cpu(X);
T = local_extract_cpu(T);

[puschIndices, ~] = nrPUSCHIndices(simParameters.Carrier, simParameters.PUSCH);
F = size(X, 1);
S = size(X, 2);
B = size(T, 2);
NSlots = size(X, 4);

targetBits = zeros(F, S, B, NSlots, "single");
dataMask = zeros(F, S, 1, NSlots, "single");
baseMask = zeros(F, S, "single");
baseMask(puschIndices) = 1;

for nslot = 1:NSlots
    dataMask(:, :, 1, nslot) = baseMask;
    for bitIdx = 1:B
        bitGrid = zeros(F, S, "single");
        bitGrid(puschIndices) = single(T(:, bitIdx, nslot));
        targetBits(:, :, bitIdx, nslot) = bitGrid;
    end
end

batch = struct();
batch.X = single(X);
batch.TargetBits = targetBits;
batch.DataMask = dataMask;
batch.SNRIn = single(snrIn);
batch.ChannelModel = string(channelModel);
batch.DelaySpread = single(delaySpread);
batch.MaxDopplerShift = single(maxDopplerShift);
batch.DMRSAdditionalPosition = int32(dmrsAdditionalPosition);
batch.DMRSConfigurationType = int32(dmrsConfigurationType);
batch.PUSCHIndices = double(puschIndices);
end

function value = local_extract_cpu(value)
if isa(value, "dlarray")
    value = extractdata(value);
end
if isa(value, "gpuArray")
    value = gather(value);
end
end
