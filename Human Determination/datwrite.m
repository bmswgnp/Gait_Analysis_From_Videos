function datwrite(filename, x, t)

nin = size(x, 2);
nout = size(t, 2);
ndata = size(x, 1);

fid = fopen(filename, 'wt');
if fid == -1
  error('Failed to open file.')
end

if size(t, 1) ~= ndata
  error('x and t must have same number of rows.');
end

fprintf(fid, ' nin   %d\n nout  %d\n ndata %d\n', nin , nout, ndata);
for i = 1 : ndata
  fprintf(fid, '%13e ', x(i,:), t(i,:));
  fprintf(fid, '\n');
end

flag = fclose(fid);
if flag == -1
  error('Failed to close file.')
end

