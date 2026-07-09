function model = hCreateTorchDeepRx(simParameters, opts)
%hCreateTorchDeepRx Create a PyTorch DeepRx network instance
%   Copyright 2025 The MathWorks, Inc.
arguments
    simParameters
    opts.Evaluate {mustBeNumericOrLogical,mustBeNonnegative} = false
end
    % Unzip the model's parameter dictionary file if no .pth is found in
    % the current directory. The example only contains the zipped model
    % parameters by default.
    if isempty(dir(fullfile('deeprx*.pth')))
        unzip(dir(fullfile('torch_trained*.zip')).name);
    end

    % Set the Python environment in MATLAB according to the options you
    % select in the main example file
    helperSetupPyenv(simParameters.PythonPath, ...
                     simParameters.PythonExecutionMode,... 
                     simParameters.PythonRequirements,...
                     Verbose=false);

    % Create a DeepRx network instance
    if isempty(simParameters.ModelPath)
        model = py.deeprx.construct_model(int16(simParameters.ModelInput), ...
                                    int16(simParameters.ModelOutput));
    else % trained network
        model = py.deeprx.construct_model(int16(simParameters.ModelInput), ...
                                    int16(simParameters.ModelOutput), ...
                                    simParameters.ModelPath);
    end

    % Move the model to a GPU if available, otherwise CPU
    model.to(py.deeprx.select_device);

    % Set the model to evaluation mode
    if opts.Evaluate
        model.eval();
    end
end