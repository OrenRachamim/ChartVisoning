"""ChartVisoning - visual machine learning on MA10 stock charts.

Pipeline:
    universe  -> pick a few hundred similar US stocks
    data      -> download daily OHLCV history (yfinance) and cache it
    charts    -> render the last 500 days of the MA10 line per stock
    dataset   -> cut every history into rolling windows, rasterize them to
                 small images, label them with the next 5/10 day move
    model     -> a small CNN that looks at the images
    train     -> fit the CNN with a strict time-based split
    predict   -> for every stock: probability of up/flat/down over the next
                 5 and 10 days + a projected MA10 path drawn on the chart
"""

__version__ = "0.1.0"
