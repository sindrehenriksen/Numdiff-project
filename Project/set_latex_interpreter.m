function set_latex_interpreter()
% Set latex interpreter as default for text, axes tick labels and legends
% in groot

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

end