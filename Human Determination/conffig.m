
function fh=conffig(y, t)

[C, rate] = confmat(y, t);

fh = figure('Name', 'Confusion matrix', ...
  'NumberTitle', 'off');

plotmat(C, 'k', 'k', 14);
title(['Classification rate: ' num2str(rate(1)) '%'], 'FontSize', 14);
