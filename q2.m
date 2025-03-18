%% =================================
%  Load Data and Define Market (Q2)
% =================================
clear; close all; clc; format short

% Toggle saving figures (1 = save, 0 = do not save)
figswitch = 0;

filename = 'datasets/Prices.xlsx'; % Directory for loading the dataset
imgDir = 'Images/';    % Directory for saving figures
txtDir = 'Results/';   % Directory for saving results
txtFilename = fullfile(txtDir, 'Q2.txt');

% Ensure directories exist
if ~exist(imgDir, 'dir'), mkdir(imgDir); end
if ~exist(txtDir, 'dir'), mkdir(txtDir); end

% Load dataset
dataset = readtable(filename, 'MissingRule', 'omitrow');
colLabels = dataset.Properties.VariableNames;
tickers = colLabels(2:end);    % Extract tickers
histPrices = dataset{:, 2:end};  % Historical prices
histPrices = flipud(histPrices);
histDates = dataset{:, 1};      % Historical dates
histDates = flipud(histDates);
[NObs, NAsset] = size(histPrices);

% Compute Asset Log-Returns
ret = log(histPrices(2:end, :) ./ histPrices(1:end-1, :));

% Split Data: In-Sample (Training) and Out-of-Sample (Test)
train = ret(1:floor(size(ret,1)/2), :);
test = ret(floor(size(ret,1)/2)+1:end, :);

%% ===========================
%  Bottom-Up Approach to Portfolio Risk (Parametric)
% ===========================
% Equally Weighted Portfolio (EQ)
w_eq = ones(NAsset, 1) / NAsset;

% Estimate mean vector and covariance matrix from training data
MeanV = mean(train)';
Sigma = cov(train);

% For parametric VaR at alpha=0.95
alpha = 0.95;
z = norminv(1 - alpha, 0, 1);

% Compute portfolio variance and VaR for the equally weighted portfolio
sg2p = w_eq' * Sigma * w_eq;
VaR_w = - z * sqrt(sg2p);

% Compute Marginal VaR (Euler’s decomposition)
MVaR_w = - z * Sigma * w_eq / sqrt(sg2p);

% Component VaR (CVaR) and percentage
CVaR_w = w_eq .* MVaR_w;
CVaR_w_p = CVaR_w / VaR_w;

% Plot Equally Weighted Comp. Var %
h = figure();
StockNames = categorical(tickers);
bar(StockNames, CVaR_w_p);
title('Component VaR (%) - Equally Weighted Portfolio');
xlabel('Asset'); ylabel('Component VaR (%)');
if figswitch, print(h, '-dpng', fullfile(imgDir, 'ComponentVaR_EW_Param.png')); end

%% ===========================
%  Portfolio Optimisation
% ===========================
% 1. Equally Weighted: w_eq (already defined)

% 2. Risk Parity (Parametric)
% Minimizes std of parametric component VaRs
x0 = w_eq;
w_rp = fmincon(@(x) std(w_rp_component(x, Sigma)), x0, ...
               [], [], ones(1, NAsset), 1, zeros(NAsset,1), ones(NAsset,1));
sg2rp = w_rp' * Sigma * w_rp;
MVaR_rp = - z * Sigma * w_rp / sqrt(sg2rp);
CVaR_rp = w_rp .* MVaR_rp;
CVaR_rp_p = CVaR_rp / sum(CVaR_rp);

% 3. Maximum Diversification (MD)
x0 = w_eq;
w_md = fmincon(@(x) - (x' * sqrt(diag(Sigma)) / sqrt(x' * Sigma * x)), x0, ...
               [], [], ones(1, NAsset), 1, zeros(NAsset,1), ones(NAsset,1));
sg2d = w_md' * Sigma * w_md;
MVaR_d = - z * Sigma * w_md / sqrt(sg2d);
CVaR_d = w_md .* MVaR_d;
CVaR_d_p = CVaR_d / sum(CVaR_d);

%% ===========================
%  Risk Parity (Non-Parametric)
% ===========================
% Minimise std of non-parametric component VaRs (via the small-window approach in notes)
alpha_np = 0.95;  % VaR confidence
epsilon_window = 0.001;  % half-window around portfolio VaR

% Define initial guess
x0 = w_eq;
% Solve with fmincon
w_rp_np = fmincon(@(x) w_rp_np_obj(x, train, alpha_np, epsilon_window), x0, ...
                  [], [], ones(1, NAsset), 1, zeros(NAsset,1), ones(NAsset,1));

% Once we have w_rp_np, we can compute its non-parametric component VaRs
% for train set:
cVaR_np_rp_vec = compute_nonparam_cVaR_vector(w_rp_np, train, alpha_np, epsilon_window);
% Sum of those cVaRs is portfolio VaR, so let's see the percentages:
VaR_rp_np = EmpiricalVaR(w_rp_np, train, alpha_np);
CVaR_rp_np_p = cVaR_np_rp_vec / VaR_rp_np;

%% Tables
% Table of portfolio weights
T_w = table(w_eq, w_rp, w_md, w_rp_np, ...
    'VariableNames', {'EQ','RP_param','MD','RP_nonparam'}, 'RowNames', tickers);
disp('Portfolio Weights (All Approaches):');
disp(T_w); 

% Component VaR for Risk Parity (Non-Parametric)
T_Component_g = table(w_eq, MVaR_w, CVaR_w, CVaR_w_p, ...
    'VariableNames', {'Weights','MVaR','CVaR','CVaR%'}, 'RowNames', tickers);
disp('Component VaR for Equally Weighted Portfolio (Parametric):');
disp(T_Component_g);

% Component VaR for Risk Parity (Parametric)
T_Component_RP = table(w_rp, MVaR_rp, CVaR_rp, CVaR_rp_p, ...
    'VariableNames', {'Weights','MVaR','CVaR','CVaR%'}, 'RowNames', tickers);
disp('Component VaR for Risk Parity Portfolio (Parametric):');
disp(T_Component_RP);

% Component VaR for Maximum Diversification Portfolio (Parametric)
T_Component_MD = table(w_md, MVaR_d, CVaR_d, CVaR_d_p, ...
    'VariableNames', {'Weights','MVaR','CVaR','CVaR%'}, 'RowNames', tickers);
disp('Component VaR for Maximum Diversification Portfolio (Parametric):');
disp(T_Component_MD);

% Component VaR for Risk Parity (Non-Parametric)
VaR_rp_np = EmpiricalVaR(w_rp_np, train, alpha_np);
CVaR_rp_np_p = cVaR_np_rp_vec / VaR_rp_np;

T_Component_RPnp = table(w_rp_np, cVaR_np_rp_vec, CVaR_rp_np_p, ...
    'VariableNames', {'Weights','CVaR','CVaR%'}, 'RowNames', tickers);
disp('Component VaR for Risk Parity Portfolio (Non-Parametric):');
disp(T_Component_RPnp);

%% ===========================
%  Out-of-Sample Performance
% ===========================
% Evaluate the 4 portfolios: EQ, RP_param, MD, RP_nonparam

% 1) Equally Weighted
portRet_EQ = test * w_eq;
Sharpe_EQ = mean(portRet_EQ) / std(portRet_EQ);
cumRet_EQ = exp(cumsum(portRet_EQ));
MaxDD_EQ = maxdrawdown(cumRet_EQ);

% 2) Risk Parity (Parametric)
portRet_RP = test * w_rp;
Sharpe_RP = mean(portRet_RP) / std(portRet_RP);
cumRet_RP = exp(cumsum(portRet_RP));
MaxDD_RP = maxdrawdown(cumRet_RP);

% 3) Maximum Diversification
portRet_MD = test * w_md;
Sharpe_MD = mean(portRet_MD) / std(portRet_MD);
cumRet_MD = exp(cumsum(portRet_MD));
MaxDD_MD = maxdrawdown(cumRet_MD);

% 4) Risk Parity (Non-Parametric)
portRet_RPnp = test * w_rp_np;
Sharpe_RPnp = mean(portRet_RPnp) / std(portRet_RPnp);
cumRet_RPnp = exp(cumsum(portRet_RPnp));
MaxDD_RPnp = maxdrawdown(cumRet_RPnp);

% Define test dates for plotting
testDates = histDates(floor(size(ret,1)/2)+2:end);

%% Compute VaR Violations (95% confidence level)
alpha_viol = 0.95;
% For parametric portfolios, compute VaR using the test return standard deviation
z_viol = norminv(1 - alpha_viol, 0, 1);

% Parametric VaR for Equally Weighted, RP (parametric) and MD portfolios:
VaR_EQ_95 = - z_viol * std(portRet_EQ);
VaR_RP_95 = - z_viol * std(portRet_RP);
VaR_MD_95 = - z_viol * std(portRet_MD);

% For the non-parametric Risk Parity portfolio, use empirical VaR:
VaR_RPnp_95 = EmpiricalVaR(w_rp_np, test, alpha_viol);

% Count violations: violation if the port. ret. is less than negative VaR.
viol_EQ   = sum(portRet_EQ < -VaR_EQ_95);
viol_RP   = sum(portRet_RP < -VaR_RP_95);
viol_MD   = sum(portRet_MD < -VaR_MD_95);
viol_RPnp = sum(portRet_RPnp < -VaR_RPnp_95);

%% Out-of-Sample Performance in Table

FinalRet_EQ   = cumRet_EQ(end);
FinalRet_RP   = cumRet_RP(end);
FinalRet_MD   = cumRet_MD(end);
FinalRet_RPnp = cumRet_RPnp(end);

T_Performance = table([Sharpe_EQ; Sharpe_RP; Sharpe_MD; Sharpe_RPnp], ...
                      [MaxDD_EQ;  MaxDD_RP; MaxDD_MD; MaxDD_RPnp], ...
                      [viol_EQ; viol_RP; viol_MD; viol_RPnp], ...
                      [FinalRet_EQ; FinalRet_RP; FinalRet_MD; FinalRet_RPnp], ...
    'VariableNames', {'SharpeRatio','MaxDrawdown','VaRViolations','FinalCumReturn'}, ...
    'RowNames', {'EQ','RP_param','MD','RP_nonparam'});
disp('Out-of-Sample Performance Metrics:');
disp(T_Performance);

%% ===========================
%  Visualisations for Out-of-Sample Performance
% ===========================
strategies = categorical({'EQ','RP\_param','MD','RP\_nonparam'});

% 1. Cumulative Returns Plot
h2 = figure();
plot(testDates, cumRet_EQ, 'LineWidth', 1.5); hold on;
plot(testDates, cumRet_RP, 'LineWidth', 1.5);
plot(testDates, cumRet_MD, 'LineWidth', 1.5);
plot(testDates, cumRet_RPnp, 'LineWidth', 1.5);
title('Cumulative Returns - Out-of-Sample');
xlabel('Date'); ylabel('Cumulative Return');
legend('EQ', 'RP\_param', 'MD', 'RP\_nonparam','Location','best');
if figswitch, print(h2, '-dpng', fullfile(imgDir, 'CumulativeReturns.png')); end

% 2. Sharpe Ratios Bar Chart
h3 = figure();
bar(strategies, [Sharpe_EQ, Sharpe_RP, Sharpe_MD, Sharpe_RPnp]);
title('Sharpe Ratios Comparison');
xlabel('Portfolio Strategy'); ylabel('Annualized Sharpe Ratio');
if figswitch, print(h3, '-dpng', fullfile(imgDir, 'SharpeRatios.png')); end

% 3. Maximum Drawdowns Bar Chart
h4 = figure();
bar(strategies, [MaxDD_EQ, MaxDD_RP, MaxDD_MD, MaxDD_RPnp]);
title('Maximum Drawdown Comparison');
xlabel('Portfolio Strategy'); ylabel('Maximum Drawdown');
if figswitch, print(h4, '-dpng', fullfile(imgDir, 'MaxDrawdowns.png')); end

% 4. VaR Violations Bar Chart (95% Confidence)
h5 = figure();
bar(strategies, [viol_EQ, viol_RP, viol_MD, viol_RPnp]);
title('Number of VaR Violations (95% Confidence)');
xlabel('Portfolio Strategy'); ylabel('Violations Count');
if figswitch, print(h5, '-dpng', fullfile(imgDir, 'VaRViolations.png')); end

%% ================
%  Functions
% ================
function mdd = maxdrawdown(cumReturns)
    peak = cumReturns(1);
    mdd = 0;
    for i = 1:length(cumReturns)
        if cumReturns(i) > peak
            peak = cumReturns(i);
        else
            dd = (cumReturns(i) - peak) / peak;
            if dd < mdd
                mdd = dd;
            end
        end
    end
end

function compVaR = w_rp_component(w, Sigma)
    % Parametric risk parity component VaRs
    portVar = w' * Sigma * w;
    z = norminv(0.01, 0, 1);
    margVaR = - z * Sigma * w / sqrt(portVar);
    compVaR = w .* margVaR;
end

function VaR_empirical = EmpiricalVaR(w, R, alpha)
    % Historical simulation VaR
    portRet = R * w;
    VaR_empirical = - prctile(portRet, (1 - alpha)*100);
end

function cVaR_vec = compute_nonparam_cVaR_vector(w, R, alpha, epsilon)
    % "Small window" method around the portfolio VaR
    portRet = R*w;
    VaR_p = - prctile(portRet, (1 - alpha)*100);
    lowBound = -VaR_p - epsilon;
    upBound  = -VaR_p + epsilon;
    idx = find(portRet >= lowBound & portRet <= upBound);
    if isempty(idx)
        cVaR_vec = zeros(length(w),1);
        return;
    end
    condMean = mean(R(idx,:), 1);   % 1xN
    margVaR = - condMean;           % 1xN
    cVaR_vec = w(:) .* margVaR(:);
end

function objVal = w_rp_np_obj(w, R, alpha, epsilon)
    % Non-parametric risk parity objective: minimize std of cVaR_i
    cVaR_vec = compute_nonparam_cVaR_vector(w, R, alpha, epsilon);
    objVal   = std(cVaR_vec);
end