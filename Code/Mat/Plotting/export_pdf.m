%%%

clear;

dir_root = 'C:\Users\paulg\Documents\PhD\Dissertation\Code\Matlab';

dir_figs = fullfile(dir_root,'Figures');

% files = dir('*.fig');
files = dir(fullfile(dir_figs,'Risk*_SSP.fig'));

N_files = numel(files);
for ii = 1:N_files
    
    fprintf('Exporting figure %i/%i ... \n',ii,N_files);
    
    filename = fullfile(files(ii).folder,files(ii).name);
    export_fig(filename,'-pdf','-transparent')
%     export_fig(filename,'-pdf')
    
end

