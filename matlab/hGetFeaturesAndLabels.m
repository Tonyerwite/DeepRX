function [results,X,T,simParameters] = hGetFeaturesAndLabels(simParameters,randParameters,opts)
    %hGetFeaturesAndLabels Generate the training features and labels
    %   This function generates the training features (X) and labels (T)
    %   matrices for training, otherwise it will return the simulation
    %   results for both the conventional and deep convolutional
    %   receivers.
    %
    %   Copyright 2024-2025 MathWorks, Inc.

    arguments
        simParameters
        randParameters
        opts.Iteration = 1
        opts.Net = []
        opts.IndependentChannel = false
    end

    % Determine the type of receiver: AI-native or conventional
    if ~isempty(opts.Net)
        opts.UseNeuralReceiver = true;
        if isa(opts.Net, "py.deeprx_model.DeepRx")
            % PyTorch based AI-native receiver
            opts.UsePyTorchNeuralReceiver = true;
            opts.Net = hCreateTorchDeepRx(simParameters, Evaluate=true);
        else
            % Deep learning toolbox based AI-native receiver
            opts.UsePyTorchNeuralReceiver = false;
        end
    else
        opts.UseNeuralReceiver = false;
    end

    % -----------------------------------------------------
    % Overwrite additional simulation parameters based on the current
    % configuration
    % -----------------------------------------------------
    simParameters = hGetAdditionalSystemParameters(simParameters, DisplayParameterSummary=false);

    % -----------------------------------------------------
    % Additional parameters based on the simulation mode
    % -----------------------------------------------------
    if simParameters.TrainNow
        simParameters.PUSCHExtension.EnableHARQ     = false; % Disable retransmissions for each process, using RV sequence [0,2,3,1]
        simParameters.PUSCHExtension.NHARQProcesses = 1;
        simParameters.PerfectChannelEstimator       = false;
    else % simulation
        simParameters.PUSCH.Modulation              = simParameters.ModulationType; % 'pi/2-BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'
        simParameters.PUSCHExtension.EnableHARQ     = false; % Disable retransmissions for each process, using RV sequence [0,2,3,1]
        simParameters.PUSCHExtension.NHARQProcesses = 1; % Number of parallel HARQ processes to use
    end
    simParameters.PUSCH.DMRS.DMRSAdditionalPosition = randParameters.DMRSAddPos; % Additional DM-RS symbol positions (max range 0...3)
    simParameters.PUSCH.DMRS.DMRSConfigurationType  = randParameters.DMRSConfigType; % DM-RS configuration type (1,2)
    simParameters.DelayProfile                      = randParameters.ChannelModel; 
    simParameters.DelaySpread                       = randParameters.DelaySpread;
    simParameters.MaximumDopplerShift               = randParameters.MaxDopplerShift;
    simParameters.SNRIn                             = randParameters.SNRIn;
    NSlots                                          = simParameters.NFrames*simParameters.Carrier.SlotsPerFrame;
    
    % -----------------------------------------------------
    % Preallocate the output parameters for memory efficiency
    % -----------------------------------------------------
    if simParameters.TrainNow
        results = [];
        X = zeros([simParameters.InputSize NSlots],'single');
        [~,tmpPUSCHIndicesInfo] = nrPUSCHIndices(simParameters.Carrier, simParameters.PUSCH);
        T = zeros(tmpPUSCHIndicesInfo.Gd, simParameters.NBits, NSlots, 'single');
    else % Simulation
        X = zeros([simParameters.InputSize],'single');
        T = [];
        totalTrBlk       = 0;
        totalCodedTrBlk  = 0;
        numBlockErrors   = 0;
        numUncodedErrors = 0;
        numCodedErrors   = 0;
        maxThroughput    = 0;
        simThroughput    = 0;
    end

    % -----------------------------------------------------
    % End-to-end UL-PUSCH simulation for training/testing
    % -----------------------------------------------------
    % Get the baseband OFDM waveform parameters
    waveformInfo = nrOFDMInfo(simParameters.Carrier); % Get information about the baseband waveform after OFDM modulation step

    % Run simulation for conventional receiver (MMSE) and DeepRx
    % Propagation Channel Model Construction
    % Create the channel model object for the simulation. Both CDL and TDL channel
    % models are supported.
    % Construct the CDL or TDL channel model object
    if contains(simParameters.DelayProfile, 'CDL', 'IgnoreCase', true)
        channel = nrCDLChannel; % CDL channel object

        % Swap transmit and receive sides as the default CDL channel is
        % configured for downlink transmissions.
        swapTransmitAndReceive(channel);

        % Turn the number of antennas into antenna panel array layouts. If
        % NRxAnts is not one of (1,2,4,8,16,32,64,128,256,512,1024), its value
        % is rounded up to the nearest value in the set. If NTxAnts is not 1 or
        % even, its value is rounded up to the nearest even number.
        channel.TransmitAntennaArray.Size = simParameters.TxAntennaSize;
        channel.ReceiveAntennaArray.Size = simParameters.RxAntennaSize;

        % Configure antenna elements
        channel.TransmitAntennaArray.Element = 'isotropic';
        channel.ReceiveAntennaArray.Element = '38.901';
        channel.CarrierFrequency = simParameters.CarrierFrequency;
    else
        channel = nrTDLChannel; % TDL channel object

        % Swap transmit and receive sides as the default TDL channel is
        % configured for downlink transmissions
        swapTransmitAndReceive(channel);

        % Set the channel geometry
        channel.NumTransmitAntennas = simParameters.NTxAnts;
        channel.NumReceiveAntennas = simParameters.NRxAnts;
    end
    
    % Ensure repeatability using the random stream. Set the same generators
    % and transforms on the CPU for reproducibility
    sc = RandStream('Threefry', 'Seed', opts.Iteration);
    RandStream.setGlobalStream(sc);
    if simParameters.TrainNow || opts.IndependentChannel
        channel.RandomStream = 'Global stream';
    end
    % Preserve the paper's per-antenna 2Rx power during evaluation. Training
    % retains the MathWorks-normalized input convention used by the cache.
    channel.NormalizeChannelOutputs = simParameters.TrainNow;
    
    % Assign simulation channel parameters and waveform sample rate to the object
    channel.DelayProfile        = simParameters.DelayProfile;
    channel.DelaySpread         = simParameters.DelaySpread;
    channel.MaximumDopplerShift = simParameters.MaximumDopplerShift;
    channel.SampleRate          = waveformInfo.SampleRate;

    % Get the maximum number of delayed samples by a channel multipath
    % component. This is calculated from the channel path with the largest
    % delay and the implementation delay of the channel filter. This is
    % required later to flush the channel filter to obtain the received signal.
    chInfo = info(channel);
    maxChDelay = chInfo.MaximumChannelDelay;
    
    % Set up redundancy version (RV) sequence for all HARQ processes
    if simParameters.PUSCHExtension.EnableHARQ
        % From PUSCH demodulation requirements in RAN WG4 meeting #88bis (R4-1814062)
        rvSeq = [0 2 3 1];
    else
        % HARQ disabled - single transmission with RV=0, no retransmissions
        rvSeq = 0;
    end

    % Create UL-SCH encoder System object to perform transport channel encoding
    encodeULSCH = nrULSCH;
    encodeULSCH.MultipleHARQProcesses = true;
    encodeULSCH.TargetCodeRate = simParameters.PUSCHExtension.TargetCodeRate;

    % Create UL-SCH decoder System object to perform transport channel
    % decoding Use layered belief propagation for LDPC decoding, with half
    % the number of iterations as compared to the default for belief
    % propagation decoding
    decodeULSCH = nrULSCHDecoder;
    decodeULSCH.MultipleHARQProcesses     = true;
    decodeULSCH.TargetCodeRate            = simParameters.PUSCHExtension.TargetCodeRate;
    decodeULSCH.LDPCDecodingAlgorithm     = simParameters.PUSCHExtension.LDPCDecodingAlgorithm;
    decodeULSCH.MaximumLDPCIterationCount = simParameters.PUSCHExtension.MaximumLDPCIterationCount;

    % Take full copies of the simulation-level parameter structures so that
    % they are not PCT broadcast variables when using parfor
    simLocal = simParameters;
    waveinfoLocal = waveformInfo;

    % Take copies of channel-level parameters to simplify subsequent
    % parameter referencing
    carrier = simLocal.Carrier;
    pusch = simLocal.PUSCH;
    puschextra = simLocal.PUSCHExtension;
    decodeULSCHLocal = decodeULSCH;  % Copy of the decoder handle to help PCT classification of variable
    reset(decodeULSCHLocal);        % Reset decoder at the start of each SNR point

    % Create PUSCH object configured for the non-codebook transmission
    % scheme, used for receiver operations that are performed with respect
    % to the PUSCH layers
    if ~simParameters.TrainNow
        puschNonCodebook = pusch;
        puschNonCodebook.TransmissionScheme = 'nonCodebook';

        % Reset the channel for each SNR point in testing so that each SNR
        % point will experience the same channel realization during
        % simulation
        reset(channel);
    end

    % Prepare simulation for new SNR point
    SNRdB = simLocal.SNRIn;

    % Specify the fixed order in which we cycle through the HARQ process
    % IDs
    harqSequence = 0:puschextra.NHARQProcesses-1;

    % Initialize the state of all HARQ processes
    harqEntity = HARQEntity(harqSequence,rvSeq);

    for nslot = 0:NSlots-1
        % Update the carrier slot number for the new slot
        carrier.NSlot = nslot;

		% Calculate the transport block size for the transmission in the
		% slot Obtain PUSCH indices and related information
		[puschIndices, puschIndicesInfo] = nrPUSCHIndices(carrier, pusch);
		% Determine the number of resource blocks
		MRB = numel(pusch.PRBSet);
		% Calculate the transport block size based on modulation, layers,
		% and other parameters
		trBlkSize = nrTBS(pusch.Modulation, pusch.NumLayers, MRB, puschIndicesInfo.NREPerPRB, puschextra.TargetCodeRate, puschextra.XOverhead);

		% HARQ processing
		% If new data is available for the current process, create a new UL-SCH transport block
		if harqEntity.NewData
			% Generate a random transport block
			trBlk = randi([0 1], trBlkSize, 1, "int8");
			% Set the transport block for encoding
			setTransportBlock(encodeULSCH, trBlk, harqEntity.HARQProcessID);
			% If new data due to previous RV sequence timeout, flush
			% decoder soft buffer explicitly
			if harqEntity.SequenceTimeout
				resetSoftBuffer(decodeULSCHLocal, harqEntity.HARQProcessID);
			end
		end

		% Encode the UL-SCH transport block
		codedTrBlock = encodeULSCH(pusch.Modulation, pusch.NumLayers, ...
								   puschIndicesInfo.G, harqEntity.RedundancyVersion, harqEntity.HARQProcessID);

		% Create resource grids for a slot
		puschGrid = nrResourceGrid(carrier, simLocal.NTxAnts);
		dmrsGrid = nrResourceGrid(carrier, pusch.NumLayers);

		% PUSCH modulation, including codebook-based MIMO precoding if
		% TxScheme = 'codebook'
		puschSymbols = nrPUSCH(carrier, pusch, codedTrBlock);

		% Non-codebook-based MIMO precoding, F precodes between PUSCH
		% layers and transmit antennas
		F = eye(pusch.NumLayers, simLocal.NTxAnts);

		% Map PUSCH symbols to the resource grid
		[~, puschAntIndices] = nrExtractResources(puschIndices, puschGrid);
		puschGrid(puschAntIndices) = puschSymbols * F;

		% Implementation-specific PUSCH DM-RS MIMO precoding and mapping
		dmrsSymbols = nrPUSCHDMRS(carrier, pusch);
		dmrsIndices = nrPUSCHDMRSIndices(carrier, pusch);
		for p = 1:size(dmrsSymbols, 2)
			[~, dmrsAntIndices] = nrExtractResources(dmrsIndices(:, p), puschGrid);
			puschGrid(dmrsAntIndices) = puschGrid(dmrsAntIndices) + dmrsSymbols(:, p) * F(p, :);
		end
		dmrsGrid(dmrsIndices) = dmrsSymbols;

		% OFDM modulation
		txWaveform = nrOFDMModulate(carrier, puschGrid);

		% Pass data through the channel model. Append zeros to flush
		% channel content. These zeros account for any delay introduced in
		% the channel.
		txWaveform = [txWaveform; zeros(maxChDelay, size(txWaveform, 2))]; %#ok<AGROW>

        % Reset the channel for each slot. This generates a batch of
        % independent channel realizations in each training iteration.
        if simParameters.TrainNow
            reset(channel);
        end
		[rxWaveform, pathGains, sampleTimes] = channel(txWaveform);

		% Add AWGN
		SNR = 10^(SNRdB/10);
		if simLocal.TrainNow
			% Retain the convention used by the existing fixed training cache.
			N0 = 1/sqrt(simLocal.NRxAnts * double(waveinfoLocal.Nfft) * SNR);
			noise = N0 * randn(size(rxWaveform), "like", rxWaveform);
		else
			% Match the reference MIMO simulation: derive complex AWGN variance
			% from the realized signal power averaged over receive antennas.
			signalPowerPerRxAntenna = mean(abs(rxWaveform).^2, "all");
			noiseVariance = signalPowerPerRxAntenna / SNR;
			noiseReal = randn(size(rxWaveform), "like", real(rxWaveform));
			noiseImag = randn(size(rxWaveform), "like", real(rxWaveform));
			noise = sqrt(noiseVariance/2) * (noiseReal + 1i*noiseImag);
		end
		rxWaveform = rxWaveform + noise;

		% Perfect synchronization using channel information
		pathFilters = getPathFilters(channel);
		[offset, ~] = nrPerfectTimingEstimate(pathGains, pathFilters);
		rxWaveform = rxWaveform(1+offset:end, :);

		% Perform OFDM demodulation
		rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
    
        if simParameters.TrainNow
            % Construct the features array
            % X              : [F S 4*NRxAnts+2 NSlot] real-valued array
            % rxGrid         : [F S NRxAnts] complex array
            % dmrsGrid       : [F S 1] complex array
            % rawChanEstGrid : [F S NRxAnts] complex array
            rawChanEstGrid = rxGrid .* conj(dmrsGrid);
            
            % Concatenate real and imaginary parts of the input arrays
            X(:,:,:,nslot+1) = cat(3, real(rxGrid), imag(rxGrid), ...
                                      real(dmrsGrid), imag(dmrsGrid), ...
                                      real(rawChanEstGrid), imag(rawChanEstGrid));
    
            % Construct labels array
            % T : [F S NBits] binary array
            Qm = puschIndicesInfo.G / puschIndicesInfo.Gd / pusch.NumLayers;
            scrambled = nrPUSCHScramble(codedTrBlock, simLocal.PUSCH.NID, simLocal.PUSCH.RNTI, []);
            T(:,:, nslot+1) = reshape(scrambled, Qm, []).';
    
            % Cast dlarray for training
            X = dlarray(X, 'SSCB');
            T = dlarray(T, 'SSB');
    
            % Cast gpuArray for GPU acceleration
            if canUseGPU
                X = gpuArray(X);
                T = gpuArray(T);
            end
        else
            if opts.UseNeuralReceiver
                % Construct the features array
                % X              : [F S 4*NRxAnts+2 NSlot] real-valued array
                % rxGrid         : [F S NRxAnts] complex array
                % dmrsGrid       : [F S 1] complex array
                % rawChanEstGrid : [F S NRxAnts] complex array
                % The checkpoint was trained with normalized channel outputs.
                % Adapt only its input features; the simulated 2Rx waveform and
                % conventional receivers retain the paper's power convention.
                neuralRxGrid = rxGrid / sqrt(simLocal.NRxAnts);
                rawChanEstGrid = neuralRxGrid .* conj(dmrsGrid);
                
                % Concatenate real and imaginary parts of the input arrays
                X = cat(3, real(neuralRxGrid), imag(neuralRxGrid), ...
                           real(dmrsGrid), imag(dmrsGrid), ...
                           real(rawChanEstGrid), imag(rawChanEstGrid));
    
                % Predict LLRs using the input resource grids.
                %
                % (i) -- If it is a PyTorch network, predict the LLRs (L)
                % using the trained model (opts.Net) and test resource
                % grids (X). This option uses a GPU for predictions
                % implicitly via the predict call inside the deeprx.py
                % module if a GPU is available. Note that the trained model
                % deepRx_30k.pth has been trained for the input and output
                % sizes of [312 14 10] and 4, respectively. The operations
                % inside the predict function are as follows:
                %
                %   1) Load the trained model using model_in 2) Cast x_test
                %   to tensor and permute for PyTorch compatibility. Note
                %   that the batch dimension is 1 in this example and the
                %   trailing singleton dimensions in MATLAB are omitted,
                %   which is not the case in PyTorch:
                %       [F S 4*Nrx+2 1] -> [1 4*Nrx+2 F S]
                %   3) Predict LLRs:
                %       L [1 NBits F S]
                %   4) Permute the tensors for MATLAB compatibility:
                %       [1 NBits F S] -> [F S NBits 1]
                %
                % (ii) -- If it is a dlnetwork object, predict LLRs using
                % the DeepRx network. This option will use a GPU for
                % predictions if GPU is available. It will output the LLRs
                % (L), which is a real-valued array with the last dimension
                % as batch L [F S NBits 1]
                if opts.UsePyTorchNeuralReceiver
                    L = single(py.deeprx.predict(opts.Net, X));
                else % dlnetwork
                    if canUseGPU
                        L = predict(opts.Net, gpuArray(dlarray(X, 'SSCB')));
                    else
                        L = predict(opts.Net, dlarray(X, 'SSCB'));
                    end
                end

                % Extract utilized LLR values
                Qm = puschIndicesInfo.G / puschIndicesInfo.Gd / pusch.NumLayers;
                
                % Note that the negation comes from the difference between
                % the LLR definition in `ldpcDecode` and BCE loss
                % functions, sigmoid(-LLR) = Prob(c = 1 | y), where c and y
                % are the LDPC-encoded codeword and channel output for the
                % codeword, respectively.
                extractedLLRs = nrExtractResources(double(puschIndices), -L(:, :, 1:Qm));
                if isa(opts.Net, "py.deeprx_model.DeepRx")
                    ulschLLRs = reshape(gather(extractedLLRs.'), [], 1);
                else % dlarray
                    ulschLLRs = reshape(gather(extractdata(extractedLLRs.')), [], 1);
                end
                ulschLLRs = nrPUSCHDescramble(ulschLLRs, simLocal.PUSCH.NID, simLocal.PUSCH.RNTI, []);
            else % conventional receiver
                if simLocal.PerfectChannelEstimator
                    % Perform perfect channel estimation using path gains
                    % from the channel
                    estChannelGrid = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset, sampleTimes);
    
                    % Obtain perfect noise estimate from noise realization
                    noiseGrid = nrOFDMDemodulate(carrier, noise(1+offset:end, :));
                    noiseEst = var(noiseGrid(:));
    
                    % Apply MIMO deprecoding to estChannelGrid for each
                    % transmission layer
                    K = size(estChannelGrid, 1);
                    estChannelGrid = reshape(estChannelGrid, K * carrier.SymbolsPerSlot * simLocal.NRxAnts, simLocal.NTxAnts);
                    estChannelGrid = estChannelGrid * F.';
                    estChannelGrid = reshape(estChannelGrid, K, carrier.SymbolsPerSlot, simLocal.NRxAnts, []);
                else
                    % Practical channel estimation using the PUSCH DM-RS
                    % for each layer
                    dmrsLayerSymbols = nrPUSCHDMRS(carrier, puschNonCodebook);
                    dmrsLayerIndices = nrPUSCHDMRSIndices(carrier, puschNonCodebook);
                    [estChannelGrid, noiseEst] = nrChannelEstimate(carrier, rxGrid, dmrsLayerIndices, dmrsLayerSymbols, 'CDMLengths', pusch.DMRS.CDMLengths);
                end
    
                % Extract PUSCH resource elements from the received grid
                [puschRx, puschHest] = nrExtractResources(puschIndices, rxGrid, estChannelGrid);
    
                % Equalization using MMSE
                [puschEq, csi] = nrEqualizeMMSE(puschRx, puschHest, noiseEst);
    
                % Decode PUSCH physical channel
                [ulschLLRs, rxSymbols] = nrPUSCHDecode(carrier, puschNonCodebook, puschEq, noiseEst);
    
                % Apply channel state information (CSI) from the equalizer
                % Include the effect of transform precoding if enabled
                if pusch.TransformPrecoding
                    MSC = MRB * 12;
                    csi = nrTransformDeprecode(csi, MRB) / sqrt(MSC);
                    csi = repmat(csi((1:MSC:end).'), 1, MSC).';
                    csi = reshape(csi, size(rxSymbols));
                end
                csi = nrLayerDemap(csi);
                Qm = length(ulschLLRs) / length(rxSymbols);
                csi = reshape(repmat(csi{1}.', Qm, 1), [], 1);
                ulschLLRs = ulschLLRs .* csi;
            end
    
            % Decode the UL-SCH transport channel
            decodeULSCHLocal.TransportBlockLength = trBlkSize;
            [decbits, blkerr] = decodeULSCHLocal(ulschLLRs, pusch.Modulation, pusch.NumLayers, harqEntity.RedundancyVersion, harqEntity.HARQProcessID);
    
            % Update current process with CRC error and advance to the next
            % process
            updateAndAdvance(harqEntity, blkerr, trBlkSize, puschIndicesInfo.G);
    
            % Store values to calculate throughput for the conventional
            % receiver
            totalTrBlk       = totalTrBlk + trBlkSize;
            totalCodedTrBlk  = totalCodedTrBlk + length(codedTrBlock);
            numBlockErrors   = numBlockErrors + blkerr;
            numUncodedErrors = numUncodedErrors + sum(xor(ulschLLRs <= 0, codedTrBlock));
            numCodedErrors   = numCodedErrors + sum(xor(decbits, trBlk));
            simThroughput    = simThroughput + (~blkerr * trBlkSize);
            maxThroughput    = maxThroughput + trBlkSize;
        end
    end

    % -----------------------------------------------------
    % Keep various results in a struct if in testing mode
    % -----------------------------------------------------
    % Please note that the HARQ is disabled by default and Coded BER won't
    % produce meaningful results if HARQ is enabled manually
    if ~simParameters.TrainNow
        % Calculate Block Error Rate (BLER)
        results.BLER = numBlockErrors / NSlots;
        
        % Calculate Uncoded Bit Error Rate (BER)
        results.UncodedBER = numUncodedErrors / totalCodedTrBlk;
        
        % Calculate Coded BER
        results.CodedBER = numCodedErrors / totalTrBlk;
        
        % Calculate the simulated throughput
        results.SimThroughput = simThroughput;
        
        % Calculate the maximum throughput
        results.MaxThroughput = maxThroughput;
        
        % Calculate the percentage throughput
        results.PercThroughput = 100 * simThroughput / maxThroughput;

        % Record all results for each SNR point
        results.SNRIn = simParameters.SNRIn;
    end
end
