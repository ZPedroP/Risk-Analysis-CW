%% ===========================
%  Load Data and SWITCHES
% Adapted from Fusai (2025) Tutorial sessions for Risk Analysis at Bayes Business School
%% ===========================
rng(123)
clear; close all; clc; format shortG
tic()

% Switches: 
figswitch = 1;      % Set to 1 to save figures (they always display)
resultswitch = 1;   % Set to 1 to save analysis results (e.g., tables) to Results directory

filename = 'datasets/Prices.xlsx'; % Directory for loading the dataset
imgDir = 'Images/';   % Directory for saving figures
txtDir = 'Results/';  % Directory for saving results
txtFilename = fullfile(txtDir, 'Q1.txt');

% Ensure directories exist
if ~exist(imgDir, 'dir'), mkdir(imgDir); end
if ~exist(txtDir, 'dir'), mkdir(txtDir); end

%% Load Data
dataset = readtable(filename, 'MissingRule', 'omitrow');
colLabels = dataset.Properties.VariableNames;
tickers = colLabels(2:end); % Extract tickers
histPrices = dataset{:, 2:end}; % Historical prices
histDates = dataset{:, 1}; % Historical dates
[NObs, NAsset] = size(histPrices);
tradeDates = histDates(2:end);

% Compute Asset Log-Returns
ret = log(histPrices(2:end, :) ./ histPrices(1:end-1, :));

%% Question 1.1: Portfolio Statistics and Plots
% Build Equally Weighted Portfolio
portRet = mean(ret, 2);
mu = mean(portRet);
sigma = std(portRet);
kurt = kurtosis(portRet);
skew = skewness(portRet);
summarystats=[mu, sigma, kurt, skew, min(portRet,[],1), max(portRet,[],1)]

% Time Series Plot of Portfolio Returns
figure;
plot(tradeDates, portRet, 'LineWidth',1.5);
title('Return Series for Equally Weighted Portfolio');
xlabel('Time');
ylabel('Return');
grid on;
if figswitch, print(gcf, '-dpng', fullfile(imgDir, 'ReturnSeries_Portfolio.png')); end

% Histogram of Portfolio Returns
figure;
histogram(portRet, 100);
title('Histogram of Equally Weighted Portfolio');
xlabel('Return');
ylabel('Frequency');
grid on;
if figswitch, print(gcf, '-dpng', fullfile(imgDir, 'Histogram_Portfolio.png')); end

% Jarque-Bera Normality Test
[jb_h,jb_p,jjb_bstat,jb_critval] = jbtest(portRet)

% Rolling Mean and Standard Deviation (6-month window ~126 trading days)
windowSize = 126; 
rollingMean = movmean(portRet, windowSize);
rollingStd = movstd(portRet, windowSize);

% Plot Rolling Mean
figure;
plot(tradeDates, rollingMean, 'LineWidth',1.5);
title('Rolling Mean of Portfolio Returns');
xlabel('Time');
ylabel('Rolling Mean');
grid on;
if figswitch, print(gcf, '-dpng', fullfile(imgDir, 'RollingMean_Portfolio.png')); end

% Plot Rolling Volatility (Std Dev)
figure;
plot(tradeDates, rollingStd, 'LineWidth',1.5);
title('Rolling Volatility (Std Dev) of Portfolio Returns');
xlabel('Time');
ylabel('Rolling Std Dev');
grid on;
if figswitch, print(gcf, '-dpng', fullfile(imgDir, 'RollingVolatility_Portfolio.png')); end

% Autocorrelation of Squared Returns (Volatility Clustering)
figure;
autocorr(portRet.^2);
title('Autocorrelation of Squared Portfolio Returns');
grid on;
if figswitch, print(gcf, '-dpng', fullfile(imgDir, 'Autocorr_SquaredReturns.png')); end

% ARCH Test for Volatility Clustering
[h_arch, pValue_arch, stat_arch, arch_critval ] = archtest(portRet);
fprintf('ARCH test result: h = %d, p-value = %.4f\n', h_arch, pValue_arch);

% Cumulative Returns Calculation
cumRet = exp(cumsum(portRet)) - 1;
figure;
plot(tradeDates, cumRet, 'LineWidth',1.5);
title('Cumulative Returns for Equally Weighted Portfolio');
xlabel('Time');
ylabel('Cumulative Return');
grid on;
if figswitch, print(gcf, '-dpng', fullfile(imgDir, 'CumulativeReturns_Portfolio.png')); end

%% Question 1.2: Rolling Window VaR Estimation Setup
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

%% IMPLEMENT 6 VaR ESTIMATES:
% 1. Bootstrap VaR (Non-Parametric) with Confidence Interval
% 2. Block Bootstrap VaR (Non-Parametric) with Confidence Interval
% 3. Hybrid Block-Filtered Bootstrap VaR with Confidence Interval
% 4. Gaussian VaR (Parametric)
% 5. Student's T VaR (MM) via Method of Moments (Parametric)
% 6. EWMA VaR (RiskMetrics) (Parametric)

confLevels = [0.90, 0.99];
nConf = length(confLevels);

VaR_g = @(a, mu, sg) - (mu + sg * icdf('norm', 1 - a, 0, 1));
VaR_t = @(a, mu, sg, nu) - (mu + sg * icdf('T', 1 - a, nu));

B = 1000;         % Number of bootstrap resamples
lambda = 0.94;    % EWMA decay factor
blockLength = 20; % Block length for block bootstrap
blockLengthFB = 5; % Block length for the hybrid block-filtered bootstrap

% Preallocate arrays for each method (point estimates and CI bounds)
VaR_boot_point = zeros(nRolling, nConf);
VaR_boot_CI_lower = zeros(nRolling, nConf);
VaR_boot_CI_upper = zeros(nRolling, nConf);

VaR_block_point = zeros(nRolling, nConf);
VaR_block_CI_lower = zeros(nRolling, nConf);
VaR_block_CI_upper = zeros(nRolling, nConf);

VaR_filtered_point = zeros(nRolling, nConf);
VaR_filtered_CI_lower = zeros(nRolling, nConf);
VaR_filtered_CI_upper = zeros(nRolling, nConf);

VaR_gauss  = zeros(nRolling, nConf);
VaR_t_MM   = zeros(nRolling, nConf); % Student's T via Method of Moments
VaR_EWMA   = zeros(nRolling, nConf);

for i = 1:nRolling
    startIdx = rollingWindowStartIndices(i);
    endIdx = find(tradeDates == rollingWindowEndDates(i), 1, 'first');
    windowReturns = portRet(startIdx:endIdx);
    Nw = length(windowReturns);
    
    mu_w = mean(windowReturns);
    sigma_w = std(windowReturns);
    
    % --- Student's T VaR (MM) ---
    % Estimate degrees of freedom and adjusted scale using MM:
    nu_mm = 4 + 6/(kurtosis(windowReturns)-3);
    sigma_mm = sqrt(((nu_mm - 2) / nu_mm) * var(windowReturns));
    
    % --- Bootstrap VaR (Standard) ---
    for j = 1:nConf
        a = confLevels(j);
        bootEstimates = zeros(B, 1);
        for b = 1:B
            bootSample = datasample(windowReturns, Nw);
            bootEstimates(b) = quantile(bootSample, 1 - a);
        end
        VaR_boot_point(i, j) = -mean(bootEstimates);
        VaR_boot_CI_lower(i, j) = -prctile(bootEstimates, 97.5);
        VaR_boot_CI_upper(i, j) = -prctile(bootEstimates, 2.5);
    end
    
    % --- Block Bootstrap VaR ---
    for j = 1:nConf
        a = confLevels(j);
        blockBootEstimates = zeros(B,1);
        for b = 1:B
            numBlocks = ceil(Nw / blockLength);
            bootSample = [];
            for nb = 1:numBlocks
                startBlock = randi(Nw - blockLength + 1);
                bootBlock = windowReturns(startBlock:startBlock+blockLength-1);
                bootSample = [bootSample; bootBlock];
            end
            bootSample = bootSample(1:Nw);
            blockBootEstimates(b) = quantile(bootSample, 1 - a);
        end
        VaR_block_point(i, j) = -mean(blockBootEstimates);
        VaR_block_CI_lower(i, j) = -prctile(blockBootEstimates, 97.5);
        VaR_block_CI_upper(i, j) = -prctile(blockBootEstimates, 2.5);
    end
    
    % --- Hybrid Block-Filtered Bootstrap VaR ---
    % Fit a GARCH(1,1) model to the window
    model = garch(1,1);
    [EstModel,~,~,~] = estimate(model, windowReturns, 'Display', 'off');
    % Forecast next-day conditional variance (only one output)
    V = forecast(EstModel, 1, 'Y0', windowReturns);
    sigma_forecast = sqrt(V);
    % Infer standardized residuals from the fitted model
    stdResids = windowReturns ./ sqrt(infer(EstModel, windowReturns));
    
    for j = 1:nConf
        a = confLevels(j);
        filteredBlockEstimates = zeros(B,1);
        for b = 1:B
            % Block bootstrap on standardized residuals with block length blockLengthFB
            numBlocksFB = ceil(Nw / blockLengthFB);
            bootSampleFB = [];
            for nb = 1:numBlocksFB
                startBlock = randi(Nw - blockLengthFB + 1);
                block = stdResids(startBlock : startBlock+blockLengthFB-1);
                bootSampleFB = [bootSampleFB; block];
            end
            bootSampleFB = bootSampleFB(1:Nw);
            % For a 1-day horizon, select one residual from the bootstrapped series
            rStar = datasample(bootSampleFB, 1);
            % Reconstruct next-day simulated return
            simulatedReturn = mu_w + sigma_forecast * rStar;
            filteredBlockEstimates(b) = simulatedReturn;
        end
        VaR_filtered_point(i, j) = -quantile(filteredBlockEstimates, 1 - a);
        VaR_filtered_CI_lower(i, j) = -prctile(filteredBlockEstimates, 97.5);
        VaR_filtered_CI_upper(i, j) = -prctile(filteredBlockEstimates, 2.5);
    end
    
    % --- Gaussian VaR ---
    for j = 1:nConf
        a = confLevels(j);
        VaR_gauss(i, j) = VaR_g(a, mu_w, sigma_w);
    end
    
    % --- Student's T VaR (MM) ---
    for j = 1:nConf
        a = confLevels(j);
        VaR_t_MM(i, j) = VaR_t(a, mu_w, sigma_mm, nu_mm);
    end
    
    % --- EWMA VaR ---
    weights_EWMA = (1 - lambda) * lambda.^( (Nw-1):-1:0 );
    sigma_EWMA = sqrt(sum(weights_EWMA(:) .* windowReturns(:).^2));
    for j = 1:nConf
        a = confLevels(j);
        VaR_EWMA(i, j) = - (sigma_EWMA * icdf('norm', 1 - a, 0, 1));
    end
end

% %% Plotting VaR Estimates and Saving Figures if figswitch is true
% if figswitch
%     % Plot and save 99% VaR estimates for all methods
%     figure;
%     plot(rollingWindowEndDates, VaR_boot_point(:,2), 'LineWidth', 1.5); hold on;
%     plot(rollingWindowEndDates, VaR_block_point(:,2), 'LineWidth', 1.5);
%     plot(rollingWindowEndDates, VaR_filtered_point(:,2), 'LineWidth', 1.5);
%     plot(rollingWindowEndDates, VaR_gauss(:,2), 'LineWidth', 1.5);
%     plot(rollingWindowEndDates, VaR_t_MM(:,2), 'LineWidth', 1.5);
%     plot(rollingWindowEndDates, VaR_EWMA(:,2), 'LineWidth', 1.5);
%     title('Daily Rolling 99% VaR Estimates');
%     xlabel('Date');
%     ylabel('VaR (Loss)');
%     legend('Bootstrap','Block Bootstrap','Hybrid Block-Filtered',...
%            'Gaussian','Student T (MM)','EWMA (RiskMetrics)','Location', 'Best');
%     grid on;
%     print(gcf, '-dpng', fullfile(imgDir, 'VaR_Estimates_99.png'));
% 
%     % Plot and save 90% VaR estimates
%     figure;
%     plot(rollingWindowEndDates, VaR_boot_point(:,1), 'LineWidth', 1.5); hold on;
%     plot(rollingWindowEndDates, VaR_block_point(:,1), 'LineWidth', 1.5);
%     plot(rollingWindowEndDates, VaR_filtered_point(:,1), 'LineWidth', 1.5);
%     plot(rollingWindowEndDates, VaR_gauss(:,1), 'LineWidth', 1.5);
%     plot(rollingWindowEndDates, VaR_t_MM(:,1), 'LineWidth', 1.5);
%     plot(rollingWindowEndDates, VaR_EWMA(:,1), 'LineWidth', 1.5);
%     title('Daily Rolling 90% VaR Estimates');
%     xlabel('Date');
%     ylabel('VaR (Loss)');
%     legend('Bootstrap','Block Bootstrap','Hybrid Block-Filtered',...
%            'Gaussian','Student T (MM)','EWMA (RiskMetrics)','Location', 'Best');
%     grid on;
%     print(gcf, '-dpng', fullfile(imgDir, 'VaR_Estimates_90.png'));
% end

%%
% Alternative plots

if figswitch
    % Define colors for parametric methods
    color_gauss = [0.5 0 0.5];   % Purple
    color_tMM   = [0 0.8 0];      % Green
    color_EWMA  = [0.5 0.7 1];    % Light Blue

    % For 99% VaR estimates - Parametric Approaches
    figure;
    plot(rollingWindowEndDates, VaR_gauss(:,2), 'LineWidth', 1.5, 'Color', color_gauss); hold on;
    plot(rollingWindowEndDates, VaR_t_MM(:,2), 'LineWidth', 1.5, 'Color', color_tMM);
    plot(rollingWindowEndDates, VaR_EWMA(:,2), 'LineWidth', 1.5, 'Color', color_EWMA);
    title('Daily Rolling 99% VaR Estimates (Parametric Approaches)');
    xlabel('Date');
    ylabel('VaR (Loss)');
    legend('Gaussian', 'Student T (MM)', 'EWMA (RiskMetrics)', 'Location', 'Best');
    grid on;
    print(gcf, '-dpng', fullfile(imgDir, 'VaR_Parametric_99.png'));
    
    % For 99% VaR estimates - Bootstrap Variations
    figure;
    plot(rollingWindowEndDates, VaR_boot_point(:,2), 'LineWidth', 1.5); hold on;
    plot(rollingWindowEndDates, VaR_block_point(:,2), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_filtered_point(:,2), 'LineWidth', 1.5);
    title('Daily Rolling 99% VaR Estimates (Bootstrap Variations)');
    xlabel('Date');
    ylabel('VaR (Loss)');
    legend('Bootstrap', 'Block Bootstrap', 'Hybrid Block-Filtered', 'Location', 'Best');
    grid on;
    print(gcf, '-dpng', fullfile(imgDir, 'VaR_Bootstrap_99.png'));
    
    % For 90% VaR estimates - Parametric Approaches
    figure;
    plot(rollingWindowEndDates, VaR_gauss(:,1), 'LineWidth', 1.5, 'Color', color_gauss); hold on;
    plot(rollingWindowEndDates, VaR_t_MM(:,1), 'LineWidth', 1.5, 'Color', color_tMM);
    plot(rollingWindowEndDates, VaR_EWMA(:,1), 'LineWidth', 1.5, 'Color', color_EWMA);
    title('Daily Rolling 90% VaR Estimates (Parametric Approaches)');
    xlabel('Date');
    ylabel('VaR (Loss)');
    legend('Gaussian', 'Student T (MM)', 'EWMA (RiskMetrics)', 'Location', 'Best');
    grid on;
    print(gcf, '-dpng', fullfile(imgDir, 'VaR_Parametric_90.png'));
    
    % For 90% VaR estimates - Bootstrap Variations
    figure;
    plot(rollingWindowEndDates, VaR_boot_point(:,1), 'LineWidth', 1.5); hold on;
    plot(rollingWindowEndDates, VaR_block_point(:,1), 'LineWidth', 1.5);
    plot(rollingWindowEndDates, VaR_filtered_point(:,1), 'LineWidth', 1.5);
    title('Daily Rolling 90% VaR Estimates (Bootstrap Variations)');
    xlabel('Date');
    ylabel('VaR (Loss)');
    legend('Bootstrap', 'Block Bootstrap', 'Hybrid Block-Filtered', 'Location', 'Best');
    grid on;
    print(gcf, '-dpng', fullfile(imgDir, 'VaR_Bootstrap_90.png'));
end













%%
%% Question 1.3: Backtesting - Count VaR Violations
% Forecast on day x evaluates return on day x+1.
VaR_models = {VaR_boot_point, VaR_block_point, VaR_filtered_point, ...
              VaR_gauss, VaR_t_MM, VaR_EWMA};
model_names = {'Bootstrap','Block Bootstrap','Hybrid Block-Filtered',...
               'Gaussian','Student T (MM)','EWMA'};

nForecasts = nRolling - 1;
nextDayReturns = portRet(idx_first+1:end);
violationCounts = zeros(length(VaR_models), 2);

for m = 1:length(VaR_models)
    for c = 1:2  % c = 1 for 90% and c = 2 for 99%
        forecastVaR = VaR_models{m}(1:nForecasts, c);
        % Change here: compare the magnitude of the loss (-return) to VaR
        violationCounts(m, c) = sum(-nextDayReturns > forecastVaR);
    end
end

varNames = {'Violations_90', 'Violations_99'};
T = array2table(violationCounts, 'RowNames', model_names, 'VariableNames', varNames);
disp(T);
if resultswitch
    writetable(T, fullfile(txtDir, 'VaR_Violation_Counts.txt'), 'WriteRowNames', true);
end

% Plot and save violation counts
figure;
bar(violationCounts);
set(gca, 'XTickLabel', model_names);
legend({'90% VaR', '99% VaR'}, 'Location', 'Best');
title('Number of VaR Violations by Model');
ylabel('Number of Violations');
grid on;
if figswitch, print(gcf, '-dpng', fullfile(imgDir, 'ViolationCounts.png')); end

%% Q1.4a Backtesting Tests - Kupiec and Conditional Coverage
% For each VaR model and each confidence level, compute:
% - Kupiec's Proportion of Failures (POF) test statistic using a stable log formulation
% - Christoffersen's Conditional Coverage test statistic

% Number of forecasts:
T_forecasts = nForecasts;  % from Q1.3, forecasts = nRolling - 1

% Define a small epsilon to avoid log(0)
epsilon = 1e-6;

% Preallocate matrices to store test statistics and p-values
LR_pof_mat = zeros(length(VaR_models), nConf);
pval_pof_mat = zeros(length(VaR_models), nConf);
LR_cc_mat = zeros(length(VaR_models), nConf);
pval_cc_mat = zeros(length(VaR_models), nConf);

fprintf('\nBacktesting Results:\n');
for m = 1:length(VaR_models)
    for c = 1:nConf
        % Set target violation probability (alpha)
        alpha = 1 - confLevels(c);  % e.g., for 99% VaR, alpha = 0.01
        
        % Get the VaR forecasts for this model (for T_forecasts days)
        forecastVaR = VaR_models{m}(1:T_forecasts, c);
        
        % Create binary violation indicator (1 if violation, 0 otherwise)
        violations = -nextDayReturns > forecastVaR;
        x = sum(violations);  % number of observed violations
        
        % Observed violation rate
        pi_hat = x / T_forecasts;
        
        % --- Kupiec's POF Test (Stable Log Formulation) ---
        % Standard formula:
        % LR_pof = -2 * [ (T - x)*log((1-alpha)/(1-pi_hat)) + x*log(alpha/pi_hat) ]
        LR_pof = -2 * ( (T_forecasts - x)*log((1 - alpha)/(1 - pi_hat + epsilon)) + ...
                         x*log(alpha/(pi_hat + epsilon)) );
        pval_pof = 1 - chi2cdf(LR_pof, 1);
        
        LR_pof_mat(m, c) = LR_pof;
        pval_pof_mat(m, c) = pval_pof;
        
        % --- Conditional Coverage: Independence Test ---
        % Build binary transition counts from the violation series v
        v = violations;
        n00 = sum(v(1:end-1)==0 & v(2:end)==0);
        n01 = sum(v(1:end-1)==0 & v(2:end)==1);
        n10 = sum(v(1:end-1)==1 & v(2:end)==0);
        n11 = sum(v(1:end-1)==1 & v(2:end)==1);
        T_trans = T_forecasts - 1;
        
        % Compute transition probabilities safely
        if (n00+n01) == 0
            pi_0 = epsilon;
        else
            pi_0 = n01 / (n00+n01);
        end
        
        if (n10+n11) == 0
            pi_1 = epsilon;
        else
            pi_1 = n11 / (n10+n11);
        end
        
        % Overall transition probability:
        pi_bar = (n01+n11) / T_trans;
        
        % Compute log likelihoods under the null and alternative
        logL0 = (n00+n10)*log(1 - pi_bar + epsilon) + (n01+n11)*log(pi_bar + epsilon);
        logL1 = n00*log(1 - pi_0 + epsilon) + n01*log(pi_0 + epsilon) + ...
                n10*log(1 - pi_1 + epsilon) + n11*log(pi_1 + epsilon);
        LR_indep = -2 * (logL0 - logL1);
        
        % Conditional Coverage test statistic
        LR_cc = LR_pof + LR_indep;
        pval_cc = 1 - chi2cdf(LR_cc, 2);  % 2 degrees of freedom
        
        LR_cc_mat(m, c) = LR_cc;
        pval_cc_mat(m, c) = pval_cc;
        
        % Display results for this model and confidence level
        fprintf('Model: %s, Confidence Level: %.0f%%\n', model_names{m}, confLevels(c)*100);
        fprintf('  Kupiec LR_pof = %.4f, p-value = %.4f\n', LR_pof, pval_pof);
        fprintf('  Conditional Coverage LR_cc = %.4f, p-value = %.4f\n\n', LR_cc, pval_cc);
    end
end

% Optionally, save these results if resultswitch is on:
if resultswitch
    TestResults = table(LR_pof_mat, pval_pof_mat, LR_cc_mat, pval_cc_mat, ...
                        'VariableNames', {'LR_pof','pval_pof','LR_cc','pval_cc'}, ...
                        'RowNames', model_names);
    disp(TestResults);
    writetable(TestResults, fullfile(txtDir, 'Backtesting_Results.txt'), 'WriteRowNames', true);
end

%% Q1.4b Distributional Backtesting for All VaR Estimates
% For each forecast date and for each VaR model, compute the PIT value.
% We assume that for nonparametric models (models 1-3) the forecast distribution is given by the 
% empirical distribution of windowReturns.

numModels = length(VaR_models);
PIT_all = cell(numModels,1);  % To store PIT values for each model

for m = 1:numModels
    PIT_model = zeros(nForecasts,1);
    for i = 1:nForecasts
        % Get the window for forecast i
        startIdx = rollingWindowStartIndices(i);
        endIdx = find(tradeDates == rollingWindowEndDates(i), 1, 'first');
        windowReturns = portRet(startIdx:endIdx);
        r_next = nextDayReturns(i);
        
        if m <= 3
            % Nonparametric models: use empirical CDF from windowReturns
            PIT_model(i) = sum(windowReturns <= r_next) / length(windowReturns);
        elseif strcmp(model_names{m}, 'Gaussian')
            % Gaussian model: use normcdf with parameters from windowReturns
            mu_w = mean(windowReturns);
            sigma_w = std(windowReturns);
            PIT_model(i) = normcdf(r_next, mu_w, sigma_w);
        elseif strcmp(model_names{m}, 'Student T (MM)')
            % Student T (MM): estimate parameters from windowReturns
            mu_w = mean(windowReturns);
            nu_mm = 4 + 6/(kurtosis(windowReturns)-3);
            sigma_mm = sqrt(((nu_mm - 2) / nu_mm) * var(windowReturns));
            % Standardize and use tcdf (MATLAB’s tcdf assumes zero location and unit scale)
            PIT_model(i) = tcdf((r_next - mu_w)/sigma_mm, nu_mm);
        elseif strcmp(model_names{m}, 'EWMA')
            % EWMA model: use EWMA volatility and window mean
            Nw = length(windowReturns);
            weights_EWMA = (1 - lambda) * lambda.^( (Nw-1):-1:0 );
            sigma_EWMA = sqrt(sum(weights_EWMA(:) .* windowReturns(:).^2));
            mu_w = mean(windowReturns);
            PIT_model(i) = normcdf(r_next, mu_w, sigma_EWMA);
        end
    end
    PIT_all{m} = PIT_model;
end

% Now, for each model, plot histograms (full and left tail) and perform KS tests.
for m = 1:numModels
    % Full distribution
    figure;
    histogram(PIT_all{m}, 20, 'Normalization', 'pdf');
    title(['Histogram of PIT Values for ' model_names{m}]);
    xlabel('PIT Value');
    ylabel('Density');
    grid on;
    
    [h_full, p_full] = kstest(PIT_all{m}, 'CDF', makedist('Uniform'));
    fprintf('KS test for full PIT for %s: h = %d, p-value = %.4f\n', model_names{m}, h_full, p_full);
    
    % Left tail: PIT values below 0.10 (10%)
    PIT_left = PIT_all{m}(PIT_all{m} < 0.10);
    figure;
    histogram(PIT_left, 10, 'Normalization', 'pdf');
    title(['Histogram of Left-Tail PIT Values for ' model_names{m}]);
    xlabel('PIT Value');
    ylabel('Density');
    grid on;
    
    [h_left, p_left] = kstest(PIT_left, 'CDF', makedist('Uniform'));
    fprintf('KS test for left-tail PIT for %s: h = %d, p-value = %.4f\n', model_names{m}, h_left, p_left);
end

%%
% Collect KS test results into arrays
modelNames = model_names(:);
numModels = length(modelNames);
KS_full_h = zeros(numModels,1);
KS_full_p = zeros(numModels,1);
KS_left_h = zeros(numModels,1);
KS_left_p = zeros(numModels,1);

for m = 1:numModels
    % KS test for full PIT values for this model
    [h_full, p_full] = kstest(PIT_all{m}, 'CDF', makedist('Uniform'));
    % KS test for left-tail PIT values (< 0.10) for this model
    PIT_left = PIT_all{m}(PIT_all{m} < 0.10);
    [h_left, p_left] = kstest(PIT_left, 'CDF', makedist('Uniform'));
    
    KS_full_h(m) = h_full;
    KS_full_p(m) = p_full;
    KS_left_h(m) = h_left;
    KS_left_p(m) = p_left;
end

% Create a table of the KS test results
KS_results = table(modelNames, KS_full_h, KS_full_p, KS_left_h, KS_left_p, ...
    'VariableNames', {'Model','KS_Full_h','KS_Full_p','KS_Left_h','KS_Left_p'});
disp(KS_results);

if resultswitch, writetable(KS_results, fullfile(txtDir, 'Backtesting_Results.txt'), 'WriteRowNames', true); end
toc()