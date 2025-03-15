%%
% TO DO: Bootstrapping is VERY VERY SLOW. We currently run 1000 samples.
% Smaller sample paths and/or alernative implementation may be necessary 
% Elapsed time is 99.981202 seconds to line 257

%% ===========================
%  Load Data and FIGURE SWITCH
% This code cell is adapted from Fusai (2025) Tutorial sessions for the
% course in Risk Analysis at Bayes Business School
% ===========================
clear; close all; clc; format long

figswitch=1 % Set figswitch=1 to show figures, else set figswitch=0.


imgDir = 'Images/'; % Directory for saving figures
txtDir = 'Results/'; % Directory for saving results
dataDir = 'datasets/Prices.xlsx'; % Directory for loading the dataset
txtFilename = fullfile(txtDir, 'Q1.txt');

% Ensure directories exist
if ~exist(imgDir, 'dir'), mkdir(imgDir); end
if ~exist(txtDir, 'dir'), mkdir(txtDir); end

% Load dataset
dataset = readtable(dataDir, 'MissingRule', 'omitrow');
colLabels = dataset.Properties.VariableNames;
tickers = colLabels(2:end); % Extract tickers
histPrices = dataset{:, 2:end}; % Historical prices
histPrices = flipud(histPrices);
histDates = dataset{:, 1}; % Historical dates
histDates = flipud(histDates);
[NObs, NAsset] = size(histPrices);
tradeDates = histDates(2:end);

% Compute Asset Log-Returns
ret = log(histPrices(2:end, :) ./ histPrices(1:end-1, :));

%% Question 1.1
% Build Equally Weighted Portfolio
portRet = mean(ret, 2);

mu = mean(portRet);
sigma = std(portRet);
kurt = kurtosis(portRet);
skew = skewness(portRet);

% Time Series Plot and Histogram of Return Dispersion
if figswitch==1  
    figure
    plot(tradeDates, portRet)
    title('Return Series for Equally Weighted Portfolio')
    xlabel('Time')
    ylabel('Return')
    grid on

    figure
    histogram(portRet, 100)
    title('Histogram of Equally Weighted Portfolio')
    xlabel('Return')
    ylabel('Frequency')
    grid on
end

% Jarque-Bera Normality Test
jb = jbtest(portRet)

% Compute Rolling Mean and STD
windowSize = 126; % 126 days ?= 6 months. So estimating 6 month rolling moments
rollingMean = movmean(portRet, windowSize);
rollingStd = movstd(portRet, windowSize);


% Plot Rolling Mean and Rolling Volatility
if figswitch==1 
    figure
    plot(tradeDates, rollingMean)
    title('Rolling Mean of Portfolio Returns')
    xlabel('Time')
    ylabel('Rolling Mean')
    grid on

    figure
    plot(tradeDates, rollingStd)
    title('Rolling Volatility (Std Dev) of Portfolio Returns')
    xlabel('Time')
    ylabel('Rolling Std Dev')
    grid on
end

% Volatility Clustering Analysis
if figswitch==1
    figure
    autocorr(portRet.^2)
    title('Autocorrelation of Squared Portfolio Returns')
    grid on
end

% ARCH Test for Volatility Clustering
[h_arch, pValue_arch] = archtest(portRet);
fprintf('ARCH test result: h = %d, p-value = %.4f\n', h_arch, pValue_arch);

% Cumulative Returns Calculation
cumRet = exp(cumsum(portRet)) - 1;

% Plot Cumulative Returns
if figswitch==1
    figure
    plot(tradeDates, cumRet)
    title('Cumulative Returns for Equally Weighted Portfolio')
    xlabel('Time')
    ylabel('Cumulative Return')
    grid on
end
%% Question 1.2
% Rolling Window Setup for 6-Month Windows (Daily)
% The first window's endpoint is 01/07/14
% For each day from that date onward, we use the previous 6 calendar months
% of data. If the ideal window start is not a trading day, we use the next 
% available day in the dataset.

firstWindowEnd = datetime(2014,7,1);
idx_first = find(tradeDates >= firstWindowEnd, 1, 'first');
nRolling = length(tradeDates) - idx_first + 1;

rollingWindowStartIndices = zeros(nRolling, 1);
rollingWindowStartDates = datetime.empty(nRolling, 0);
rollingWindowEndDates   = datetime.empty(nRolling, 0);

for idx = idx_first:length(tradeDates)
    currentDate = tradeDates(idx);
    idealWindowStart = currentDate - calmonths(6);
    windowStartIdx = find(tradeDates >= idealWindowStart, 1, 'first');
    rollingWindowStartIndices(idx - idx_first + 1) = windowStartIdx;
    rollingWindowStartDates(idx - idx_first + 1) = tradeDates(windowStartIdx);
    rollingWindowEndDates(idx - idx_first + 1) = currentDate;
end

rollingWindowsTable = table(rollingWindowStartDates, rollingWindowEndDates, ...
    'VariableNames', {'WindowStart', 'WindowEnd'});
disp(rollingWindowsTable);

%% IMPLEMENT 6 VaR ESTIMATES
% 1. Empirical VaR (non-parametric)
% 2. Bootstrap VaR (non-parametric)
% 3. Gaussian VaR (parametric)
% 4. Student's T VaR (via Max. Likelihood) (parametric)
% 5. Student's T VaR (via Method of Moments) (parametric)
% 6. Conditional VaR using EWMA (RiskMetrics) (parametric)

confLevels = [0.90, 0.99];
nConf = length(confLevels);

VaR_g = @(a, mu, sg) - (mu + sg * icdf('norm', 1 - a, 0, 1));
VaR_t = @(a, mu, sg, nu) - (mu + sg * icdf('T', 1 - a, nu));
VaR_np = @(alpha, Returns) -prctile(Returns, (1-alpha)*100);

B = 1000;         % Number of bootstrap resamples
lambda = 0.94;    % EWMA decay factor. This is JPM's recommended factor for RiskMetrics

nRolling = length(rollingWindowEndDates);
VaR_emp    = zeros(nRolling, nConf);
VaR_boot   = zeros(nRolling, nConf);
VaR_gauss  = zeros(nRolling, nConf);
VaR_t_MLE  = zeros(nRolling, nConf);
VaR_t_MM   = zeros(nRolling, nConf);
VaR_EWMA   = zeros(nRolling, nConf);

for i = 1:nRolling
    startIdx = rollingWindowStartIndices(i);
    endIdx = find(tradeDates == rollingWindowEndDates(i), 1, 'first');
    
    windowReturns = portRet(startIdx:endIdx);
    Nw = length(windowReturns);
    
    mu_w = mean(windowReturns);
    sigma_w = std(windowReturns);
    
    % Student's T (MLE)
    pd_t = fitdist(windowReturns, 'tlocationscale');
    
    % Student's T (MM)
    mu_mm_w = mu_w;
    nu_mm_w = 4 + 6 / (kurtosis(windowReturns) - 3);
    sg_mm_w = sqrt(((nu_mm_w - 2) / nu_mm_w) * var(windowReturns));
    
    % Conditional Volatility
    % Use RiskMetrics weights without renormalisation
    weights_EWMA = (1 - lambda) * lambda.^( (Nw-1):-1:0 );
    sigma_EWMA = sqrt(sum((weights_EWMA(:) .* windowReturns(:).^2)));
    
    % Bootstrap VaR: initialize matrix for bootstrap estimates
    VaR_boot_current = zeros(B,1);
    
    for j = 1:nConf
        a = confLevels(j);
        
        % 1. Empirical VaR
        VaR_emp(i, j) = VaR_np(a, windowReturns);
        
        % 2. Bootstrap VaR
        for b = 1:B
            bootSample = datasample(windowReturns, Nw);
            VaR_boot_current(b) = quantile(bootSample, 1 - a);
        end
        VaR_boot(i, j) = -mean(VaR_boot_current);
        
        % 3. Gaussian VaR
        VaR_gauss(i, j) = VaR_g(a, mu_w, sigma_w);
        
        % 4. Student's T VaR (MLE)
        VaR_t_MLE(i, j) = VaR_t(a, pd_t.mu, pd_t.sigma, pd_t.nu);
        
        % 5. Student's T VaR (MM)
        VaR_t_MM(i, j) = VaR_t(a, mu_mm_w, sg_mm_w, nu_mm_w);
        
        % 6. Conditional VaR
        VaR_EWMA(i, j) = - (sigma_EWMA * icdf('norm', 1 - a, 0, 1));
    end
end

if figswitch==1
% Plot 99% VaR estimates
    figure;
    plot(rollingWindowEndDates, VaR_emp(:,2), 'LineWidth', 1.5); hold on;
    plot(rollingWindowEndDates, VaR_boot(:,2), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_gauss(:,2), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_t_MLE(:,2), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_t_MM(:,2), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_EWMA(:,2), 'LineWidth', 1.5);
    title('Daily Rolling 99% VaR Estimates');
    xlabel('Date');
    ylabel('VaR (Loss)');
    legend('Empirical','Bootstrap','Gaussian','Student''s T (MLE)', 'Student''s T (MM)', 'EWMA (RiskMetrics)', 'Location', 'Best');
    grid on;

    % Plot 90% VaR estimates over time
    figure;
    plot(rollingWindowEndDates, VaR_emp(:,1), 'LineWidth', 1.5); hold on;
    plot(rollingWindowEndDates, VaR_boot(:,1), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_gauss(:,1), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_t_MLE(:,1), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_t_MM(:,1), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_EWMA(:,1), 'LineWidth', 1.5);
    title('Daily Rolling 90% VaR Estimates');
    xlabel('Date');
    ylabel('VaR (Loss)');
    legend('Empirical','Bootstrap','Gaussian','Student''s T (MLE)', 'Student''s T (MM)', 'EWMA (RiskMetrics)', 'Location', 'Best');
    grid on;
end

%% Question 1.3
% ************************************************************************
% RATIONALE
% i) Forecast-Return Alignment:
% VaR forecast computed on day x (end of rolling window) evaluates return
% on day x+1.
% i.e. for each forecast date, we compare the next day’s portfolio return
% against the VaR forecast.
% ii) Counting Violations Convention:
% We count each model’s violations separately for both 90% and 99% levels
% If a return violates the 99% VaR, it also violates 90% VaR by def'n
% iii) Data Alignment:
% portRet is vector. Combined with dates in tradeDates, pairs returns with
% dates (i.e. simpler to lookup for the “next day” return)
% iv) Final Forecast:
% No data after final day so this won't contribute to VaR
% ************************************************************************

% Define models in cell array
VaR_models = {VaR_emp, VaR_boot, VaR_gauss, VaR_t_MLE, VaR_t_MM, VaR_EWMA};
model_names = {'Empirical', 'Bootstrap', 'Gaussian', 'Student T (MLE)', 'Student T (MM)', 'EWMA'};

% Number of forecasts available for evaluation (exclude the last forecast)
nForecasts = nRolling - 1;

% All possible next day returns
nextDayReturns = portRet(idx_first+1:end);

% Initialise violation counts matrix
violationCounts = zeros(6, 2);

% Loop over models and confidence levels
for m = 1:length(VaR_models)
    for c = 1:2  % c=1: 90% VaR, c=2: 99% VaR
        % Get the VaR forecasts for this model, excluding the last forecast (no next day available)
        forecastVaR = VaR_models{m}(1:nForecasts, c);
        
        % A violation occurs when the actual return is less than the forecast.
        % Note: if the 99% forecast is violated (i.e. return is even lower), it also
        % counts as a violation for the 90% level. Here we count each model/confidence separately.
        violationCounts(m, c) = sum(nextDayReturns < forecastVaR);
    end
end

% Create a table to display the violation counts nicely
varNames = {'Violations_90', 'Violations_99'};
T = array2table(violationCounts, 'RowNames', model_names, 'VariableNames', varNames);
disp(T);

% Plot the violation counts in a grouped bar chart
if figswitch==1
    figure;
    bar(violationCounts);
    set(gca, 'XTickLabel', model_names);
    legend({'90% VaR', '99% VaR'}, 'Location', 'Best');
    title('Number of VaR Violations by Model');
    ylabel('Number of Violations');
    grid on;
end