clear; close all; clc;
rng(123);

%% 1) Load Data and Define Market Parameters
% Define file and folder names for inputs and outputs
dataFile = 'datasets/Prices.xlsx'; % Excel file with historical asset prices
marketName = 'Top 6 Tech Stocks';  % Market name (for display or labeling purposes)
figuresDir = 'Images/';            % Directory to save generated figures
resultsDir = 'Results/';           % Directory to save any result files

% Create output directories if they do not exist
if ~exist(figuresDir, 'dir'), mkdir(figuresDir); end
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% Read the dataset; omit rows with missing values
dataTable = readtable(dataFile, 'MissingRule', 'omitrow');

% Extract variable names and separate dates from asset prices
variableNames = dataTable.Properties.VariableNames;
assetTickers = variableNames(2:end); % Assume first column holds dates
prices = dataTable{:, 2:end}; % Historical asset prices
prices = flipud(prices); % Flip to ensure chronological order

dates = dataTable{:, 1}; % Historical dates
dates = flipud(dates);

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
numBootstrap = 10000; % Number of bootstrap iterations

% Preallocate matrix for storing bootstrap probability estimates
bootstrapProb = zeros(numBootstrap, horizonDays); % Bootstrap probability estimates

% Define the loss threshold using continuous compounding.
% A 5% loss corresponds to log(0.95) (≈ -0.0513)
lossThreshold = log(0.95);

% Run bootstrap simulations
for i = 1:numBootstrap
    % Generate random indices for simulating return paths
    % Each column represents a simulated path over 'horizonDays' using 'numPeriods' available returns
    randomIndices = randi(numPeriods, numPeriods, horizonDays);
    
    % Simulate cumulative portfolio log-returns along each bootstrap path.
    % For each simulated path (each row), compute the cumulative sum over days.
    simulatedCumulativeReturns = cumsum(portfolioLogReturns(randomIndices), 2);
    
    % For each day in the horizon, compute the proportion of simulated paths 
    % where the cumulative return falls below the loss threshold.
    % Note: Divide by the number of simulated paths (numPeriods) per bootstrap iteration.
    for day = 1:horizonDays
        bootstrapProb(i, day) = nnz(simulatedCumulativeReturns(:, day) < lossThreshold) / numPeriods;
    end
end

% Average the bootstrap probability estimates over all bootstrap iterations
avgBootstrapProbability = mean(bootstrapProb);

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