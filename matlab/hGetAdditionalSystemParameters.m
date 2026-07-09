function simParameters = hGetAdditionalSystemParameters(simParameters, opts)
    % hGetAdditionalSystemParameters Get additional parameters needed in PUSCH simulations
    %   This function configures additional system parameters required for
    %   Physical Uplink Shared Channel (PUSCH) simulations.
    %
    %   Copyright 2024 MathWorks, Inc.

    arguments
        simParameters
        opts.DisplayParameterSummary = true; % Option to display the parameter summary
    end

    % -----------------------------------------------------
    % Main simulation parameter settings
    % -----------------------------------------------------
    % Set waveform type and PUSCH numerology (Subcarrier Spacing and CP type)
    simParameters.Carrier = nrCarrierConfig; % Carrier resource grid configuration
    simParameters.Carrier.NSizeGrid = simParameters.NSizeGrid; % Bandwidth in number of resource blocks (e.g., 52 RBs at 15 kHz SCS for 10 MHz BW)
    simParameters.Carrier.SubcarrierSpacing = simParameters.SubcarrierSpacing; % Subcarrier Spacing (15, 30, 60, 120 kHz)
    simParameters.Carrier.CyclicPrefix = 'Normal'; % Cyclic Prefix type ('Normal' or 'Extended')
    simParameters.Carrier.NCellID = 0; % Cell identity

    % PUSCH and UL-SCH parameters
    simParameters.PUSCH = nrPUSCHConfig; % Base configuration for all PUSCH transmissions in the BLER simulation
    simParameters.PUSCHExtension = struct(); % Structure for additional UL-SCH and PUSCH simulation parameters

    % Define PUSCH time-frequency resource allocation per slot (full grid)
    simParameters.PUSCH.PRBSet = 0:simParameters.Carrier.NSizeGrid-1; % PUSCH PRB allocation
    simParameters.PUSCH.SymbolAllocation = [0, simParameters.Carrier.SymbolsPerSlot]; % PUSCH symbol allocation in each slot
    simParameters.PUSCH.MappingType = 'A'; % PUSCH mapping type ('A': slot-wise, 'B': non slot-wise)

    % Scrambling identifiers
    simParameters.PUSCH.NID = simParameters.Carrier.NCellID;
    simParameters.PUSCH.RNTI = 1;

    % Define transform precoding, layering, and transmission scheme
    simParameters.PUSCH.TransformPrecoding = false; % Enable/disable transform precoding
    simParameters.PUSCH.NumLayers = 1; % Number of PUSCH transmission layers
    simParameters.PUSCH.TransmissionScheme = 'nonCodebook'; % Transmission scheme ('nonCodebook', 'codebook')
    simParameters.PUSCH.NumAntennaPorts = 1; % Number of antenna ports for codebook-based precoding
    simParameters.PUSCH.TPMI = 0; % Precoding matrix indicator for codebook-based precoding

    % Define codeword modulation
    simParameters.PUSCH.Modulation = simParameters.ModulationType; % Modulation type ('pi/2-BPSK', 'QPSK', '16QAM', '64QAM', '256QAM')

    % PUSCH DM-RS configuration
    simParameters.PUSCH.DMRS.DMRSTypeAPosition = 2; % Mapping type A only. First DM-RS symbol position (2,3)
    simParameters.PUSCH.DMRS.DMRSLength = 1; % Number of front-loaded DM-RS symbols (1: single symbol, 2: double symbol)
    simParameters.PUSCH.DMRS.NumCDMGroupsWithoutData = 2; % Number of CDM groups without data
    simParameters.PUSCH.DMRS.NIDNSCID = 0; % Scrambling identity (0...65535)
    simParameters.PUSCH.DMRS.NSCID = 0; % Scrambling initialization (0,1)
    simParameters.PUSCH.DMRS.NRSID = 0; % Scrambling ID for low-PAPR sequences (0...1007)
    simParameters.PUSCH.DMRS.GroupHopping = 0; % Group hopping (0,1)
    simParameters.PUSCH.DMRS.SequenceHopping = 0; % Sequence hopping (0,1)

    % -----------------------------------------------------
    % Additional simulation and UL-SCH related parameters
    % -----------------------------------------------------
    % Target code rate
    simParameters.PUSCHExtension.TargetCodeRate = simParameters.CodeRate; % Code rate for calculating transport block size

    % HARQ process and rate matching/TBS parameters
    simParameters.PUSCHExtension.XOverhead = 0; % Set PUSCH rate matching overhead for TBS (Xoh)

    % LDPC decoder parameters
    simParameters.PUSCHExtension.LDPCDecodingAlgorithm = 'Normalized min-sum'; % LDPC decoding algorithm
    simParameters.PUSCHExtension.MaximumLDPCIterationCount = 6; % Maximum number of LDPC iterations

    % Define overall transmission antenna geometry at end-points
    % M:  Number of rows in each antenna panel
    % N:  Number of columns in each antenna panel
    % P:  Number of polarizations (1 or 2)
    % Mg: Number of rows in the array of panels
    % Ng: Number of columns in the array of panels
    simParameters.TxAntennaSize = [1, 1, 1, 1, 1]; % Transmitter antenna configuration [M N P Mg Ng]
    simParameters.RxAntennaSize = [1, 1, 2, 1, 1]; % Receiver antenna configuration [M N P Mg Ng]
    simParameters.NTxAnts = prod(simParameters.TxAntennaSize); % Total number of transmit antennas
    simParameters.NRxAnts = prod(simParameters.RxAntennaSize); % Total number of receive antennas

    % -----------------------------------------------------
    % AI-based receiver network parameters
    % -----------------------------------------------------
    % Input image size: [F S 4*Nr+2]
    [~, info, ~] = nrPUSCHIndices(simParameters.Carrier, simParameters.PUSCH);
    simParameters.NBits = info.G / info.Gd; % Number of bits
    simParameters.InputSize = [simParameters.Carrier.NSizeGrid * 12, simParameters.Carrier.SymbolsPerSlot, 4 * simParameters.NRxAnts + 2]; % Input size
    simParameters.OutputSize = [simParameters.Carrier.NSizeGrid * 12, simParameters.Carrier.SymbolsPerSlot, simParameters.NBits]; % Output size

    % Display the parameter summary if enabled
    if opts.DisplayParameterSummary
        fprintf('\n%s\n', repmat('_', 1, 40));
        fprintf('\nSimulation summary\n');
        fprintf('%s\n', repmat('_', 1, 40));
        fprintf('\n');
        fprintf('- %-33s [ %s ]\n', 'Input (X) size:', num2str(simParameters.InputSize));
        fprintf('- %-33s [ %s ]\n', 'Output (L) size:', num2str(simParameters.OutputSize));
        fprintf('\n');
        fprintf('- %-33s %-5d\n', 'Number of subcarriers (F):', simParameters.Carrier.NSizeGrid * 12);
        fprintf('- %-33s %-5d\n', 'Number of symbols (S):', simParameters.Carrier.SymbolsPerSlot);
        fprintf('- %-33s %-5d\n', 'Number of receive antennas (Nrx):', simParameters.NRxAnts);
    end
end