classdef helperLibraryChecker < handle

%   Copyright 2024-2025 The MathWorks, Inc.
  properties (Access = private)
    CheckedLibraries
    LastExecutionMode
    FileTimestamps
  end

  methods (Access = private)
    function obj = helperLibraryChecker()
      obj.CheckedLibraries = containers.Map('KeyType', 'char', 'ValueType', 'logical');
      obj.FileTimestamps = containers.Map('KeyType', 'char', 'ValueType', 'double');
    end
  end

  methods (Static)
    function singleObj = getInstance()
      persistent uniqueInstance
      if isempty(uniqueInstance)
        uniqueInstance = helperLibraryChecker();
      end
      singleObj = uniqueInstance;
    end
  end

  methods
    function status = isLibraryChecked(obj, libraryName)
      if isKey(obj.CheckedLibraries, libraryName)
        status = obj.CheckedLibraries(libraryName);
      else
        status = false;
      end
    end

    function setLibraryChecked(obj, libraryName, status)
      obj.CheckedLibraries(libraryName) = status;
    end

    function mode = getLastExecutionMode(obj)
      mode = obj.LastExecutionMode;
    end

    function setLastExecutionMode(obj, mode)
      obj.LastExecutionMode = mode;
    end

    function timestamp = getFileTimestamp(obj, fileName)
      if isKey(obj.FileTimestamps, fileName)
        timestamp = obj.FileTimestamps(fileName);
      else
        timestamp = NaN;
      end
    end

    function setFileTimestamp(obj, fileName, timestamp)
      obj.FileTimestamps(fileName) = timestamp;
    end

    function status = isFileTimestampStored(obj, fileName)
      status = isKey(obj.FileTimestamps, fileName);
    end
  end
end