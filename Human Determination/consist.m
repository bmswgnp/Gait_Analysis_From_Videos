function errstring = consist(model, type, inputs, outputs)

% Assume that all is OK as default
errstring = '';

% If type string is not empty
if ~isempty(type)
  % First check that model has type field
  if ~isfield(model, 'type')
    errstring = 'Data structure does not contain type field';
    return
  end
  % Check that model has the correct type
  s = model.type;
  if ~strcmp(s, type)
    errstring = ['Model type ''', s, ''' does not match expected type ''',...
	type, ''''];
    return
  end
end

% If inputs are present, check that they have correct dimension
if nargin > 2
  if ~isfield(model, 'nin')
    errstring = 'Data structure does not contain nin field';
    return
  end

  data_nin = size(inputs, 2);
  if model.nin ~= data_nin
    errstring = ['Dimension of inputs ', num2str(data_nin), ...
	' does not match number of model inputs ', num2str(model.nin)];
    return
  end
end

% If outputs are present, check that they have correct dimension
if nargin > 3
  if ~isfield(model, 'nout')
    errstring = 'Data structure does not conatin nout field';
    return
  end
  data_nout = size(outputs, 2);
  if model.nout ~= data_nout
    errstring = ['Dimension of outputs ', num2str(data_nout), ...
	' does not match number of model outputs ', num2str(model.nout)];
    return
  end

% Also check that number of data points in inputs and outputs is the same
  num_in = size(inputs, 1);
  num_out = size(outputs, 1);
  if num_in ~= num_out
    errstring = ['Number of input patterns ', num2str(num_in), ...
	' does not match number of output patterns ', num2str(num_out)];
    return
  end
end
