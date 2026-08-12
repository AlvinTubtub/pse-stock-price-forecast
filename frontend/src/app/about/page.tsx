export default function AboutPage() {
  return (
    <div className="animate-[fadeIn_0.3s_ease-out] max-w-5xl mx-auto space-y-12">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white tracking-tight mb-2">
          About the Research
        </h1>
        <p className="text-lg text-slate-400">
          BSIT Data Analytics Capstone Project Overview
        </p>
      </div>

      <div className="space-y-8">
        {/* Project Overview */}
        <section className="bg-dark-card border border-dark-border rounded-2xl p-8">
          <h2 className="text-2xl font-bold text-white mb-4 border-b border-dark-border pb-2">
            Project Overview
          </h2>

          <h3 className="text-lg font-semibold text-brand-400 mt-6 mb-2">The Problem</h3>
          <p className="text-slate-300 leading-relaxed text-sm mb-6">
            Existing local studies on Philippine stock forecasting often focus on a single model or only track market indices. Furthermore, the outputs are usually code-heavy and difficult for non-programmers to interpret. There is also a lack of comparative analysis to prove which model architecture works best for specific industries, as no single forecasting model performs perfectly across all market conditions.
          </p>

          <h3 className="text-lg font-semibold text-brand-400 mb-2">How the System Works</h3>
          <p className="text-slate-300 leading-relaxed text-sm">
            Historical OHLCV data (Open, High, Low, Close, and Volume) spanning several years are sourced primarily from the Official PSE Daily Quotations Reports. The dataset is then cleaned and preprocessed through data validation, feature engineering, lag feature generation, and scaling to prepare it for forecasting. The processed data are chronologically divided into an 85% Development Dataset and a 15% Hold-out Test Dataset, with Rolling-Origin Validation performed within the Development Dataset to preserve the temporal order of observations and prevent data leakage. Three forecasting models—Lag-Informed Regression, ARIMA, and LSTM—are subsequently developed, trained, and evaluated under identical experimental conditions. Each model predicts the next-day price change, which is then reconstructed into the predicted next-day closing price presented in the dashboard. Finally, the ForecastPH web dashboard visualizes the forecasting results, model performance, and comparative analyses through an interactive interface for educational and analytical purposes.
          </p>
        </section>

        {/* Methodology & Evaluation */}
        <section className="bg-dark-card border border-dark-border rounded-2xl p-8">
          <h2 className="text-2xl font-bold text-white mb-4 border-b border-dark-border pb-2">
            Methodology &amp; Evaluation
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
            <div>
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">
                Target Users
              </h3>
              <ul className="space-y-2.5 text-slate-300 text-sm">
                <li className="flex items-center gap-2">
                  <span className="text-brand-400 font-bold">✓</span> Budding Traders (Primary Users)
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-brand-400 font-bold">✓</span> Students and Learning Users
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-brand-400 font-bold">✓</span> Researchers and Data Analytics Developers
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-brand-400 font-bold">✓</span> Academic Community
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-brand-400 font-bold">✓</span> Future Researchers
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">
                Statistical Rigor
              </h3>
              <p className="text-slate-300 leading-relaxed text-sm mb-3">
                RMSE (Root Mean Square Error) serves as the primary evaluation metric due to its sensitivity to large errors.
              </p>
              <p className="text-slate-300 leading-relaxed text-sm bg-dark-bg p-4 rounded-lg border border-dark-border">
                <strong className="text-white">Significance Testing:</strong> To ensure that the performance differences between models (Lag Reg vs ARIMA vs LSTM) are not just due to random chance, <strong className="text-white">Diebold-Mariano (DM) tests within companies and stock-level Friedman tests across companies</strong> are applied.
              </p>
            </div>
          </div>
        </section>

        {/* Responsible Use Disclaimer */}
        <section className="bg-gradient-to-br from-dark-card to-dark-bg border border-dark-border rounded-2xl p-8 text-center relative overflow-hidden">
          <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-brand-500 to-transparent"></div>
          <h2 className="text-xl font-bold text-white mb-4">Responsible Use Disclaimer</h2>
          <p className="text-slate-400 text-sm leading-relaxed max-w-3xl mx-auto">
            This dashboard is strictly an <strong className="text-slate-200">educational and analytical decision-support tool</strong>. It relies solely on historical market data and does not account for breaking news, economic shocks, geopolitical events, or unexpected market anomalies. The forecasts, model rankings, and insights provided do not constitute financial advice, buy-or-sell recommendations, automated trading signals, or guaranteed investment outcomes. Users should conduct their own research and consult qualified financial professionals before making investment decisions.
          </p>
        </section>
      </div>
    </div>
  );
}
