export default function LearnPage() {
  return (
    <div className="animate-[fadeIn_0.3s_ease-out] max-w-5xl mx-auto space-y-12">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white tracking-tight mb-4">
          Understanding the Forecasts
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto">
          A beginner-friendly guide to the data, models, and metrics used in this capstone research.
        </p>
      </div>

      <div className="space-y-8">
        {/* Data Fundamentals: OHLCV */}
        <div className="bg-dark-card border border-dark-border rounded-3xl p-8 md:p-10 relative overflow-hidden">
          <div className="relative z-10">
            <h2 className="text-2xl font-bold text-brand-400 mb-4 flex items-center gap-2">
              📊 What is OHLCV?
            </h2>
            <p className="text-slate-300 leading-relaxed mb-6">
              Before we can forecast the future, we have to look at the past. Our models learn purely from numerical historical data, which are the basic building blocks of daily stock market records. We do not use news sentiment, rumors, or economic indicators.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4">
              <div className="bg-dark-bg p-4 rounded-xl border border-dark-border text-center">
                <div className="text-brand-400 font-bold text-2xl mb-1">O</div>
                <div className="text-white text-sm font-semibold">Open</div>
                <div className="text-xs text-slate-500 mt-1">Price when market opens.</div>
              </div>

              <div className="bg-dark-bg p-4 rounded-xl border border-dark-border text-center">
                <div className="text-green-400 font-bold text-2xl mb-1">H</div>
                <div className="text-white text-sm font-semibold">High</div>
                <div className="text-xs text-slate-500 mt-1">Highest price of the day.</div>
              </div>

              <div className="bg-dark-bg p-4 rounded-xl border border-dark-border text-center">
                <div className="text-red-400 font-bold text-2xl mb-1">L</div>
                <div className="text-white text-sm font-semibold">Low</div>
                <div className="text-xs text-slate-500 mt-1">Lowest price of the day.</div>
              </div>

              <div className="bg-brand-900/40 p-4 rounded-xl border border-brand-500/50 text-center shadow-[0_0_15px_rgba(59,130,246,0.15)] relative">
                <div className="absolute -top-2.5 -right-2 bg-brand-500 text-[9px] font-bold text-white px-2 py-0.5 rounded uppercase">
                  TARGET
                </div>
                <div className="text-brand-300 font-bold text-2xl mb-1">C</div>
                <div className="text-white text-sm font-semibold">Close</div>
                <div className="text-xs text-slate-300 mt-1">Final price. What we predict.</div>
              </div>

              <div className="bg-dark-bg p-4 rounded-xl border border-dark-border text-center">
                <div className="text-purple-400 font-bold text-2xl mb-1">V</div>
                <div className="text-white text-sm font-semibold">Volume</div>
                <div className="text-xs text-slate-500 mt-1">Total shares traded.</div>
              </div>
            </div>
          </div>
        </div>

        {/* Models Section */}
        <div>
          <h2 className="text-2xl font-bold text-white text-center mt-12 mb-6">
            The Forecasting Models
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Lag Regression */}
            <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:-translate-y-1 transition-transform">
              <div className="w-12 h-12 bg-slate-800 rounded-xl flex items-center justify-center text-slate-300 mb-4 border border-slate-700 text-xl font-bold">
                🧮
              </div>
              <h3 className="text-xl font-bold text-white mb-1">Lag Regression</h3>
              <p className="text-sm font-medium text-slate-400 mb-4">&quot;The Pattern Spotter&quot;</p>
              <p className="text-sm text-slate-300 leading-relaxed">
                An interpretable machine learning model. It studies recent historical prices and trading volumes, uses PACF to identify meaningful lag relationships (patterns repeating over days), and LASSO regularization to retain only the strongest predictors.
              </p>
            </div>

            {/* ARIMA */}
            <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:-translate-y-1 transition-transform">
              <div className="w-12 h-12 bg-blue-900/30 rounded-xl flex items-center justify-center text-blue-400 mb-4 border border-blue-900 text-xl font-bold">
                📈
              </div>
              <h3 className="text-xl font-bold text-white mb-1">ARIMA</h3>
              <p className="text-sm font-medium text-blue-400 mb-4">&quot;The Trend Tracker&quot;</p>
              <p className="text-sm text-slate-300 leading-relaxed">
                A traditional statistical time-series standard. It analyzes historical closing prices, underlying trends, differencing (to stabilize data), and past forecasting errors to mathematically estimate the next day&apos;s closing price.
              </p>
            </div>

            {/* LSTM */}
            <div className="bg-dark-card border border-brand-500/50 rounded-2xl p-6 hover:-translate-y-1 transition-transform shadow-[0_0_20px_rgba(59,130,246,0.1)] relative overflow-hidden">
              <div className="absolute top-0 right-0 bg-brand-500 text-white text-[10px] font-bold px-3 py-1 rounded-bl-xl">
                DEEP LEARNING
              </div>
              <div className="w-12 h-12 bg-brand-900/50 rounded-xl flex items-center justify-center text-brand-300 mb-4 border border-brand-500/50 text-xl font-bold">
                🧠
              </div>
              <h3 className="text-xl font-bold text-white mb-1">LSTM</h3>
              <p className="text-sm font-medium text-brand-400 mb-4">&quot;The Deep Thinker&quot;</p>
              <p className="text-sm text-slate-300 leading-relaxed">
                A deep learning recurrent neural network. Long Short-Term Memory (LSTM) studies sequences of historical prices. It explicitly learns to &quot;remember&quot; important long-term patterns and &quot;forget&quot; irrelevant noise to generate forecasts.
              </p>
            </div>
          </div>
        </div>

        {/* Metrics Section */}
        <div className="pt-8">
          <h2 className="text-2xl font-bold text-white text-center mb-4">
            The Metrics: How Do We Know if a Model is Good?
          </h2>
          <p className="text-slate-400 text-center max-w-3xl mx-auto mb-8 text-sm leading-relaxed">
            When you look at the Model Performance page, you will see a few different scores. We use these to grade the models by comparing what the model predicted against what the stock&apos;s price actually was on that day.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* RMSE */}
            <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:border-blue-500/50 transition-colors">
              <h3 className="text-lg font-bold text-blue-400 mb-1">
                RMSE (Root Mean Squared Error)
              </h3>
              <p className="text-sm font-medium text-white mb-3">Our Main Grading Score</p>
              <p className="text-sm text-slate-400 leading-relaxed mb-4">
                Think of this as the average prediction error in raw Pesos, but it heavily penalizes huge misses. If a model gets most days right but is completely wrong on one day, its RMSE will look bad.
              </p>
              <div className="inline-block px-3 py-1 bg-green-500/10 text-green-400 border border-green-500/20 rounded text-xs font-bold uppercase tracking-wider">
                Lower is better
              </div>
            </div>

            {/* MAE */}
            <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:border-green-500/50 transition-colors">
              <h3 className="text-lg font-bold text-green-400 mb-1">
                MAE (Mean Absolute Error)
              </h3>
              <p className="text-sm font-medium text-white mb-3">The Average Miss</p>
              <p className="text-sm text-slate-400 leading-relaxed mb-4">
                This is simply the average difference between the predicted price and the actual price in Pesos. If the MAE is ₱2.00, it means the model&apos;s forecasts were off by an average of 2 Pesos.
              </p>
              <div className="inline-block px-3 py-1 bg-green-500/10 text-green-400 border border-green-500/20 rounded text-xs font-bold uppercase tracking-wider">
                Lower is better
              </div>
            </div>

            {/* MASE */}
            <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:border-yellow-500/50 transition-colors">
              <h3 className="text-lg font-bold text-yellow-400 mb-1">
                MASE (Mean Absolute Scaled Error)
              </h3>
              <p className="text-sm font-medium text-white mb-3">The Scale-Free Miss</p>
              <p className="text-sm text-slate-400 leading-relaxed mb-4">
                This compares the model&apos;s error against a naive benchmark (like guessing yesterday&apos;s price). If MASE is below 1.0, the model is doing better than just blindly guessing the trend. Because it&apos;s a ratio, we can fairly compare models across cheap and expensive stocks.
              </p>
              <div className="inline-block px-3 py-1 bg-green-500/10 text-green-400 border border-green-500/20 rounded text-xs font-bold uppercase tracking-wider">
                Lower is better
              </div>
            </div>

            {/* R2 */}
            <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:border-purple-500/50 transition-colors">
              <h3 className="text-lg font-bold text-purple-400 mb-1">R² (R-squared)</h3>
              <p className="text-sm font-medium text-white mb-3">The Explanation Score</p>
              <p className="text-sm text-slate-400 leading-relaxed mb-4">
                This tells us what percentage of the stock&apos;s actual movement the model successfully captured. Higher is better (a score closer to 1.0 or 100% means the model fits the data very well).
              </p>
              <div className="inline-block px-3 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded text-xs font-bold uppercase tracking-wider">
                Higher is better
              </div>
            </div>
          </div>
        </div>

        {/* Warning / Disclaimer Card */}
        <div className="mt-12 bg-red-950/30 border border-red-900/50 rounded-2xl p-6 flex flex-col sm:flex-row items-center sm:items-start gap-4">
          <div className="text-red-500 text-3xl shrink-0">⚠️</div>
          <div>
            <h3 className="text-lg font-bold text-red-400 mb-2">
              Forecasts Are Not Financial Advice
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              This dashboard is strictly an educational and analytical decision-support tool. It relies solely on historical market data and does not account for breaking news, economic shocks, geopolitical events, or unexpected market anomalies. The forecasts, model rankings, and insights provided do not constitute financial advice, buy-or-sell recommendations, automated trading signals, or guaranteed investment outcomes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
