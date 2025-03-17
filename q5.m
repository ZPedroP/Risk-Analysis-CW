clear; close all; clc;

%**************************************************%
%%%%%%%% BOTTOM-UP APPROACH TO PORTFOLIO RISK %%%%%%
%%%%%%%% BOOTSTRAP SIMULATION N-PERIODS     %%%%%%%%%
%**************************************************%

%% ============================
%  Load Data and Define Market
% ============================
dataDir = 'datasets/Prices.xlsx'; % Directory for loading the dataset
marketName = 'Top 6 Tech Stocks'; % Market name
imgDir = 'Images/'; % Directory for saving figures
txtDir = 'Results/'; % Directory for saving results
txtFilename = fullfile(txtDir, 'Bootstrap_Multiperiod.txt'); % Output file for results

% Ensure directories exist
if ~exist(imgDir, 'dir'), mkdir(imgDir); end
if ~exist(txtDir, 'dir'), mkdir(txtDir); end

% Load dataset
dataset = readtable(dataDir, 'MissingRule', 'omitrow'); % Read data
colLabels = dataset.Properties.VariableNames; % Column labels
tickers = colLabels(2:end); % Extract tickers (asset names)

histPrices = dataset{:, 2:end}; % Historical prices
histPrices = flipud(histPrices);

histDates = dataset{:, 1}; % Historical dates
histDates = flipud(histDates);

[NObs, NAsset] = size(histPrices); % Number of observations and assets

timeAxis = histDates(2:end);

% Compute Asset Log-Returns
logRet = price2ret(histPrices);
T = size(logRet, 1); % Number of time periods

weights = ones(NAsset, 1)/NAsset; % Equally weighted portfolio
portLogRet = logRet * weights; % Portfolio log-returns

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Bootstrap Estimates for N-Days VaR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ndays = 50; % VaR horizon (in days)
Nb = 10000; % Number of bootstrap samples

% Preallocate arrays for bootstrap results
VaRNdays = zeros(Nb, Ndays); % Bootstrap VaR values for each horizon
ESNdays = zeros(Nb, Ndays); % Bootstrap ES values for each horizon
ProbNdays = zeros(Nb, Ndays); % Bootstrap Prob values for each horizon

% Repeat Bootstrap simulations
for i = 1:Nb
    % Simulate T x Ndays cumulative returns
    U = randi(T, T, Ndays); % Random indices for bootstrapping
    simLogRetT = cumsum(portLogRet(U), 2); % Cumulative returns for each horizon
    
    % Compute the probability of losing more than 5% at different horizons
    for nday = 1:Ndays
        ProbNdays(i, nday) = nnz(simLogRetT(:, nday) < -0.05)/Nb;
    end
end

% Bootstrap probabilities
ProbBdays = mean(ProbNdays);

%% ============================
%  Plot Bootstrap Results
% ============================
% Plot Prob over the horizon
h1 = figure('Color', [1 1 1]);
plot(1:Ndays, ProbBdays, 'g*', 'LineWidth', 1.5); % Prob estimates

xlabel('Horizon (days)', 'Interpreter', 'latex');
title('Bootstrap Estimates of n-days Probabilities', 'Interpreter', 'latex');
legend('Probability (< 5%)', 'Location', 'best', 'Interpreter', 'latex');
saveas(h1, fullfile(imgDir, 'Q5_Prob_ndays.png'));
