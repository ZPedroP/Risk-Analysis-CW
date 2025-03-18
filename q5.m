clear; close all; clc;
rng(123);

%% 1) Load Data and Define Market Parameters
% Define file and folder names for inputs and outputs
dataFile = 'datasets/Prices.xlsx'; % Excel file with historical asset prices
marketName = 'Top 6 Tech Stocks';  % Market name (for display or labeling purposes)
figuresDir = 'images/';            % Directory to save generated figures
resultsDir = 'results/';           % Directory to save any result files

% Create output directories if they do not exist
if ~exist(figuresDir, 'dir'), mkdir(figuresDir); end
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% Read the dataset; omit rows with missing values
dataTable = readtable(dataFile, 'MissingRule', 'omitrow');

% Extract variable names and separate dates from asset prices
variableNames = dataTable.Properties.VariableNames;
assetTickers = variableNames(2:end); % Assume first column holds dates
prices = dataTable{:, 2:end}; % Historical asset prices

dates = dataTable{:, 1}; % Historical dates

% Get the number of observations and assets
[numObs, numAssets] = size(prices);

% Compute asset log-returns using MATLAB's price2ret function
assetLogReturns = price2ret(prices); % Each column corresponds to an asset
numPeriods = size(assetLogReturns, 1); % Number of return periods

% Define an equally weighted portfolio
portfolioWeights = ones(numAssets, 1) / numAssets;
portfolioLogReturns = assetLogReturns * portfolioWeights; % Portfolio log-returns

%% 2) Bootstrap Simulation for Multi-Day Cumulative Returns
% Set simulation parameters
horizonDays = 50; % Time horizon for risk estimation (in days)
numBootstrap = 1000; % Number of bootstrap iterations

% Preallocate matrix for storing bootstrap probability estimates
bootstrapProb = zeros(numBootstrap, horizonDays); % Bootstrap probability estimates

% Define the loss threshold using continuous compounding.
% A 5% loss corresponds to log(0.95) (≈ -0.0513)
lossThreshold = log(0.95);

% Generate a matrix of random indices with dimensions [numPaths, horizonDays]
% Each index is chosen uniformly from 1 to numPeriods, representing a random selection 
% of historical daily returns for each simulated path over the specified horizon.
randomIndices = randi(numPeriods, [numBootstrap, horizonDays]);

% Compute the cumulative sum of portfolio log-returns along each simulated path.
% The cumulative sum is taken across the columns (i.e., over the days).
simulatedCumulative = cumsum(portfolioLogReturns(randomIndices), 2);

% Calculate the average probability that the cumulative return in each simulated path 
% falls below the loss threshold. This yields a vector of probabilities for each day.
avgBootstrapProbability = mean(simulatedCumulative < lossThreshold);

%% 3) Gaussian Theoretical Probability Estimate
% Under the assumption that daily portfolio log-returns are normally distributed,
% the sum of 'h' independent log-returns is normally distributed with:
%    Mean = h * (daily mean)
%    Std  = sqrt(h) * (daily standard deviation)
%
% Therefore, the probability that the h-day cumulative return is below the
% loss threshold is given by the normal CDF:
%    P[Sum < lossThreshold] = normcdf((lossThreshold - h*mu) / (sqrt(h)*sigma))
dailyMean = mean(portfolioLogReturns);
dailyStd = std(portfolioLogReturns);
daysVector = 1:horizonDays; % Vector of day horizons
gaussianProbability = normcdf((lossThreshold - daysVector * dailyMean) ./ (sqrt(daysVector) * dailyStd));

%% 4) Plot and Save the Comparison of Bootstrap and Gaussian Estimates
figureHandle = figure('Color', [1 1 1]);
plot(1:horizonDays, avgBootstrapProbability, 'g-', 'LineWidth', 1.5); % Bootstrap estimates (green stars)
hold on;
plot(1:horizonDays, gaussianProbability, 'r-', 'LineWidth', 1.5); % Gaussian estimates (red line)

% Label the plot using LaTeX interpreter for consistency
xlabel('Time Horizon (Days)', 'Interpreter', 'latex');
ylabel('Probability of $>$5\% Loss', 'Interpreter', 'latex');
title('Comparison of Bootstrap and Gaussian Estimates for $>$5\% Loss Over 50-Day Horizon', 'Interpreter', 'latex');
legend('Bootstrap Estimate', 'Gaussian Estimate', 'Location', 'best', 'Interpreter', 'latex');
grid on;

% Save the figure to the specified directory
saveas(figureHandle, fullfile(figuresDir, 'Bootstrap_vs_Gaussian_Probability.png'));

%% 5) Display Probability Estimates in a Table
% Create a table containing the horizon days along with probability estimates
resultsTable = table((1:horizonDays)', avgBootstrapProbability(:), gaussianProbability(:), ...
    'VariableNames', {'HorizonDays', 'BootstrapProbability', 'GaussianProbability'});

% Display the table in the command window
disp('Probability Estimates Table:');
disp(resultsTable);

% (Optional) Save the table to a CSV file for future reference
writetable(resultsTable, fullfile(resultsDir, 'Probability_Estimates.csv'));