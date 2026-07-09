function currentPenv = helperSetupPyenv(exePath, exeMode, reqsFileName, opts)
%   Copyright 2024-2025 The MathWorks, Inc.

arguments
    exePath {mustBeText}
    exeMode {mustBeText}
    reqsFileName {mustBeText}
    opts.Verbose = true;
end

if opts.Verbose
    disp("Setting up Python environment")
end
currentPenv = pyenv;

% Store last execution mode and file timestamps
checker = helperLibraryChecker.getInstance();

% Check if the execution environment has changed
if strcmp(string(currentPenv.ExecutionMode), "OutOfProcess") && ...
        strcmp(string(currentPenv.Status), "Loaded") && ...
        (strcmp(exeMode, "InProcess") || hasPythonFilesChanged(checker))
    % Terminate the current environment if needed
    terminate(pyenv);
    currentPenv = pyenv(Version=exePath, ExecutionMode=exeMode);
else
    if any(strcmp(string(currentPenv.Status), ["NotLoaded","Terminated"]))
        currentPenv = pyenv(Version=exePath, ExecutionMode=exeMode);
    elseif strcmp(string(currentPenv.ExecutionMode), "InProcess") && ...
            (strcmp(exeMode, "OutOfProcess") || ...
            ~strcmp(exePath, currentPenv.Executable))
        error("To run this example using OutOfProcess mode or a different Python version, restart MATLAB then rerun the example.")    
    end
end

% Update the last known execution mode
checker.setLastExecutionMode(exeMode);

% Update the file modification times
updatePythonFileTimestamps(checker);

% Verify the libraries in the reqsFileName file are installed
helperCheckInstall(reqsFileName, opts.Verbose);
end

function changed = hasPythonFilesChanged(checker)
% Get the current directory's .py files
pyFiles = dir('*.py');
changed = false;

for i = 1:length(pyFiles)
    fileName = pyFiles(i).name;
    lastModified = pyFiles(i).datenum;

    if ~checker.isFileTimestampStored(fileName) || ...
            checker.getFileTimestamp(fileName) ~= lastModified
        changed = true;
        return;
    end
end
end

function updatePythonFileTimestamps(checker)
% Get the current directory's .py files
pyFiles = dir('*.py');

for i = 1:length(pyFiles)
    fileName = pyFiles(i).name;
    lastModified = pyFiles(i).datenum;
    checker.setFileTimestamp(fileName, lastModified);
end
end

function helperCheckInstall(reqsFileName, verbose)
if verbose
    fprintf("Parsing %s \n",reqsFileName)
end

[packageNames, packageVersions] = parseRequirements(reqsFileName);

try
  py.warnings.filterwarnings("ignore", message="TypedStorage is deprecated");
catch ME
  if strcmp(ME.identifier,'MATLAB:Pyenv:PythonTerminated')
    currEnv = pyenv;
    exePath = currEnv.Executable;
    exeMode = currEnv.ExecutionMode;
    terminate(currEnv)
    pyenv(Version=exePath, ExecutionMode=exeMode);
    py.warnings.filterwarnings("ignore", message="TypedStorage is deprecated");
  end
end

if count(py.sys.path, pwd) == 0
    insert(py.sys.path, int32(0), pwd);
end

checker = helperLibraryChecker.getInstance();

for i = 1:numel(packageNames)
    if verbose
        fprintf("Checking required package '%s'\n", packageNames{i})
    end
    if checker.isLibraryChecked(packageNames{i})
        continue;
    end

    libStatus = py.helperinstalledlibs.check_library(packageNames{i});
    if ~libStatus{1}
        error("%s is not installed." + ...
            " Check the list of required libraries in %s file in the example directory.", ...
            packageNames{i},reqsFileName)
    else
        installedV = strsplit(string(libStatus{2}), '.');
        requiredV = strsplit(string(packageVersions{i}), '.');
        if ~(str2double(installedV{1}) > str2double(requiredV{1}) || ...
                str2double(installedV{1}) == str2double(requiredV{1}) && ...
                str2double(installedV{2}) >= str2double(requiredV{2}))
            error("The installed %s version is %s.%s, the required version is %s.%s", ...
                packageNames{i}, installedV{1}, installedV{2}, requiredV{1}, requiredV{2})
        end
    end

    checker.setLibraryChecked(packageNames{i}, true);
end
if verbose
    disp("Required Python libraries are installed.")
end
end

function [packageNames,packageVersions] = parseRequirements(reqsFileName)

if ~isfile(reqsFileName)
    error('Unrecognized file %s, provide a valid requirements file name.',reqsFileName)
end

% Open the file for reading
fileID = fopen(reqsFileName, 'r');

% Read the file contents into a cell array of strings
requirements = textscan(fileID, '%s', 'Delimiter', '\n');
fclose(fileID);

% Extract the cell array
requirements = requirements{1};

% Initialize cell arrays to store package names and versions
packageNames = {};
packageVersions = {};

for i = 1:length(requirements)
    line = strtrim(requirements{i});

    % Skip empty lines and comments
    if isempty(line) || startsWith(line, "#") || startsWith(line,"--")
        continue;
    end

    % Split the line by '==' to separate package name and version
    tokens = split(line, '==');

    % Store the package name
    packageNames{end+1} = tokens{1};

    % Store the package version if specified
    if length(tokens) > 1
        packageVersions{end+1} = tokens{2};
    else
        packageVersions{end+1} = '';
    end
end
end