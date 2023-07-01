import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si

import statsmodels.api as sm
import itertools
from statsmodels.tsa.stattools import coint
from statsmodels.regression.rolling import RollingOLS

import os

from dataclasses import dataclass

import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

CURWD = os.getcwd()
DATA_DIR = f"{CURWD}/DATA/"
FILE_TYPE = "csv"


@dataclass
class IOData:
    data_cache: str  # = ["csv", "parquet", "mongodb", "sqlite3"]
    data_source: str  # daloopa etc

    def __post_init__(self):
        self.data_logger = logging.getLogger()
        self.data_logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        self.data_logger.addHandler(stdout_handler)

        file_handler = logging.FileHandler("iodata.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self.data_logger.addHandler(file_handler)

    def save(self, data, data_category, data_sub_category):
        """
        data_category: str ticker
        data_subcategory: str etl data source name
        data: pd.DataFrame

        """
        self.data = data
        self.data_category = data_category
        self.data_sub_category = data_sub_category

        if not os.path.exists(f"{DATA_DIR}/{self.data_category}/"):
            os.makedirs(f"{DATA_DIR}/{self.data_category}/")

        file_name = (
            f"{DATA_DIR}/{self.data_category}/{self.data_category}_{self.data_source}"
        )
        # try:
        if self.data_sub_category != "":
            file_name = f"{file_name}_{self.data_sub_category}"

        if self.data_cache == "csv":
            file_name = file_name + "." + self.data_cache
            self.data.to_csv(file_name, index=False)

        # self.data_logger.info(f"save {file_name}")
        # except:
        #    self.data_logger.error(f"Error saving {file_name}")

    def read(self, data_category, data_sub_category=""):
        """
        data_category: str ticker
        data_subcategory: str etl data source name

        """

        self.data_category = data_category
        self.sub_category = data_sub_category

        file_name = (
            f"{DATA_DIR}/{self.data_category}/{self.data_category}_{self.data_source}"
        )
        #        try:
        file_name = file_name + "." + self.data_cache
        self.data = pd.read_csv(file_name)
        # self.data_logger.info(f"read {file_name}")
        return self.data


class Portfolio:
    def __init__(self, stock_list, log=False):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        self.logger.addHandler(stdout_handler)

        file_handler = logging.FileHandler("portfolio.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        self.stock_list = stock_list
        self.get_prices(log)
        self.get_pairs()

    def read_data(self, ticker, data_source):
        io = IOData(data_cache="csv", data_source=data_source)
        df = io.read(data_category=ticker)
        return df

    def save_data(self, df, ticker, data_source):
        io = IOData(data_cache="csv", data_source=data_source)
        io.save(df, data_category=ticker, data_sub_category="")
        return

    def get_fundamental_data(self, ticker=""):
        list_df = []
        filter_ticker = ""
        if ticker != "":
            filter_ticker = ticker

        from glob import glob

        def read_pattern(patt):
            files = glob(patt)
            # Create empty dataframe
            df = pd.DataFrame()
            for f in files:
                df = pd.concat(
                    [df, pd.read_csv(f, low_memory=False, encoding="ISO-8859-1")]
                )
            return df.reset_index(drop=True)

        df = read_pattern("*.csv")

        for ticker in self.stock_list:
            df = read_pattern(f"{CURWD}/DATA/{ticker}/{ticker}_actuals.csv")
            df = df.loc[df["PeriodType"] == "Quarterly"]
            df.columns = [x.lower() for x in df.columns]
            df["filing_date"] = df["quarterenddate"]
            df["month"] = df["quarterenddate"].str.split("/").str[0]
            df["month"] = df["month"].str.zfill(2)
            df["filing_date_ym"] = (
                df["quarterenddate"].str.split("/").str[2] + "-" + df["month"]
            )
            list_df.append(df)

        self.fundamentals = pd.concat(list_df)
        if filter_ticker != "":
            df = self.fundamentals.loc[
                self.fundamentals["ticker"] == filter_ticker
            ].copy()
            return df
        return

    def rolling_ols(self, df_regression, ticker, kpi, window=8):
        df = df_regression.loc[df_regression["ticker"] == ticker]
        df = df_regression.loc[df_regression["ltcode"] == kpi]

        df = df.copy()
        df = df.dropna()

        if len(df) < window:
            # self.logger.error(f"{ticker} {kpi} length is less than {window}")
            return None
        df["five_day_alpha_shifted"] = pd.to_numeric(df["five_day_alpha_shifted"])
        try:
            model = RollingOLS.from_formula(
                "five_day_alpha_shifted ~ numberonly", data=df, window=8
            )
            rres = model.fit()
            return pd.concat(
                [
                    df,
                    rres.params,
                    pd.DataFrame(rres.rsquared, columns=["r_squared"]),
                ],
                axis=1,
            )
        except:
            # self.logger.error(f"{ticker} {kpi} model error")
            # self.logger.error(df)
            return None

    def get_rolling_ols(self, force=True):
        filter_list = [
            "KM",
            "IS",
        ]

        df_fundamentals = self.fundamentals.copy()
        df_fundamentals = df_fundamentals[df_fundamentals["category"].isin(filter_list)]

        self.earnings_alpha["rdate"] = pd.to_datetime(self.earnings_alpha["rdate"])
        self.earnings_alpha["fdate"] = pd.to_datetime(self.earnings_alpha["fdate"])

        df_regression = pd.merge(
            df_fundamentals[
                [
                    "ticker",
                    "numberonly",
                    "filing_date",
                    "filing_date_ym",
                    "ltcode",
                    "category",
                ]
            ],
            self.earnings_alpha[
                [
                    "ticker",
                    "five_day_alpha_shifted",
                    "rdate",
                    "fdate",
                    "surprise_pct",
                    "filing_date_ym",
                ]
            ],
            how="left",
            left_on=["ticker", "filing_date_ym"],
            right_on=["ticker", "filing_date_ym"],
        )

        df_regression["numberonly"] = df_regression["numberonly"].pct_change(4)

        df_all_list = []
        tickers = df_regression["ticker"].unique()
        for ticker in tickers:
            df = None
            df_list = []

            if force == False:
                df = self.read_data(ticker=ticker, data_source="rolling_ols")

            if df is None:
                kpis = df_regression[df_regression["ticker"] == ticker][
                    "ltcode"
                ].unique()

                for kpi in kpis:
                    # we could use pandas apply here - but I leave it in a loop for debug
                    df = self.rolling_ols(df_regression, ticker, kpi)
                    if df is not None:
                        df_list.append(df)

                if len(df_list) == 0:
                    print(f"no ols data for {ticker}")
                    continue
                else:
                    df = pd.concat(df_list).dropna()
                    self.save_data(df, ticker, "rolling_ols")
            df_all_list.append(df)

        self.rolling_ols_data = pd.concat(df_all_list).dropna()

    def rolling_beta(self, returns, period):
        cov = returns.iloc[0:, 0].rolling(period).cov(returns.iloc[0:, 1])
        market_var = returns.iloc[0:, 1].rolling(period).var()
        individual_beta = cov / market_var
        return individual_beta

    def get_alphas(self):
        cols = self.price_data.columns
        alpha_cols = ["price_date"]
        df_earnings_list = []
        for col in cols:
            if "date" not in col:
                self.price_data[col + "_pct_change"] = self.price_data[col].pct_change()
                self.price_data[col + "_pct_change_5"] = self.price_data[
                    col
                ].pct_change(5)
                self.price_data[col + "_log_return"] = np.log(
                    1 + self.price_data[col + "_pct_change"]
                )
                cols = [col + "_pct_change", "SPY_pct_change", "price_date"]
                df = self.price_data[cols]
                self.price_data[col + "_60_day_rolling_beta"] = self.rolling_beta(
                    df, 60
                )
                if "SPY" not in col:
                    self.price_data[col + "_five_day_alpha"] = (
                        self.price_data[col + "_pct_change_5"]
                        - self.price_data["SPY_pct_change_5"]
                        * self.price_data[col + "_60_day_rolling_beta"]
                    )
                    self.price_data[col + "_five_day_alpha_shifted"] = self.price_data[
                        col + "_five_day_alpha"
                    ].shift(-5)
                    alpha_cols.append(col + "_five_day_alpha")
                    alpha_cols.append(col + "_five_day_alpha_shifted")

                    df_earnings = pd.read_csv(
                        f"{CURWD}/DATA/{col}/{col}_alphavantage.csv"
                    )
                    df_earnings = df_earnings[
                        ["reportedDate", "fiscalDateEnding", "surprisePercentage"]
                    ]
                    df_earnings.columns = [
                        col + "_rdate",
                        col + "_fdate",
                        col + "_surprisepct",
                    ]
                    df_earnings[col + "_rdate"] = pd.to_datetime(
                        df_earnings[col + "_rdate"]
                    )

                    self.price_data["price_date"] = pd.to_datetime(
                        self.price_data["price_date"]
                    )
                    df_earnings[col + "_rdate"] = pd.to_datetime(
                        df_earnings[col + "_rdate"]
                    )

                    df_earnings = pd.merge(
                        self.price_data[
                            ["price_date", col + "_five_day_alpha_shifted"]
                        ],
                        df_earnings,
                        how="left",
                        left_on="price_date",
                        right_on=col + "_rdate",
                    ).dropna()
                    df_earnings.to_csv(
                        f"{CURWD}/DATA/{col}/{col}_earnings_five_day_alpha.csv"
                    )

                    df_earnings.columns = [
                        "price_date",
                        "five_day_alpha_shifted",
                        "rdate",
                        "fdate",
                        "surprise_pct",
                    ]
                    df_earnings["ticker"] = col
                    df_earnings_list.append(df_earnings)

        self.earnings_alpha = pd.concat(df_earnings_list)
        self.earnings_alpha["rdate"] = self.earnings_alpha["rdate"].astype(str)
        self.earnings_alpha["filing_date_ym"] = (
            self.earnings_alpha["fdate"].str.split("-").str[0]
            + "-"
            + self.earnings_alpha["fdate"].str.split("-").str[1]
        )

        self.alphas = self.price_data[alpha_cols].dropna()

    def get_pairs(self):
        self.all_pairs = list(itertools.combinations(self.stock_list, 2))
        self.cointegrating_pairs = []
        self.pairs = []

        for item in self.all_pairs:
            pair = self.Pair(
                self.price_data[item[0]],
                self.price_data[item[1]],
                item[0],
                item[1],
                self.price_data["price_date"],
            )
            self.pairs.append(pair)

    class Pair:
        def __init__(self, x, y, x_name, y_name, price_date):
            self.x_name = x_name
            self.y_name = y_name
            self.price_date = price_date
            self.x = x
            self.y = y
            self.cointegrates = False
            self.beta = sm.OLS(x, y).fit().params[0]
            self.spread = y - self.beta * x
            self.normalized_spread_trading = (
                self.spread - self.spread.mean()
            ) / self.spread.std()
            self.coint_t, self.pvalue, self.crit_value = coint(x, y)
            if self.pvalue < 0.05:
                self.cointegrates = True
            return

        def plot_spread(self):
            df = pd.DataFrame()
            df["price_date"] = self.price_date
            df["normalized_spread"] = self.normalized_spread_trading
            df.plot(
                x="price_date",
                y="normalized_spread",
                ltcode=f"{self.x_name} vs {self.y_name}",
            )

    def get_prices(self, log=False):
        list_df = []
        for item in self.stock_list:
            df = self.get_price_data(item)
            list_df.append(df)
        self.df_long = pd.concat(list_df, axis=0)
        self.df_wide = self.df_long.pivot_table(
            index=["price_date"], columns="ticker", values="adjclose"
        ).reset_index()

        self.spy = self.get_price_data("SPY")
        self.spy = (
            self.spy.reset_index()
            .pivot_table(index="price_date", columns="ticker", values="adjclose")
            .reset_index()
        )

        self.spy["price_date"] = pd.to_datetime(self.spy["price_date"])
        self.df_wide["price_date"] = pd.to_datetime(self.df_wide["price_date"])
        self.price_data = pd.merge(
            self.spy,
            self.df_wide,
            how="inner",
            left_on="price_date",
            right_on="price_date",
        )
        self.price_data = self.price_data.dropna()
        if log == True:
            for item in self.stock_list:
                self.price_data[item] = np.log(self.price_data[item])
            self.price_data["SPY"] = np.log(self.price_data["SPY"])

    def get_price_data(self, ticker):
        file_name = f"{DATA_DIR}/{ticker}/{ticker}_yfinance.{FILE_TYPE}"
        if os.path.exists(file_name):
            df = pd.read_csv(f"{file_name}", index_col=False)
        else:
            price_data = si.get_data(ticker, start_date="01/01/2012")
            df = pd.DataFrame(price_data).reset_index()
            df.columns = [
                "price_date",
                "open",
                "high",
                "low",
                "close",
                "adjclose",
                "volume",
                "ticker",
            ]
            df["price_date"] = pd.to_datetime(df["price_date"])
            if FILE_TYPE == "csv":
                if not os.path.exists(f"{DATA_DIR}{ticker}/"):
                    os.makedirs(f"{DATA_DIR}{ticker}/")
                df.to_csv(f"{file_name}")

        # df = df[["adjclose"]]
        df["pct_change"] = df.adjclose.pct_change()
        df["log_return"] = np.log(1 + df["pct_change"].astype(float))
        df["ticker"] = ticker
        return df


class StockData:
    def get_cross_correlations(
        self,
        stock_list=[],
        rolling_window=60,
    ):
        i = []
        j = []
        run_list = []
        df_output = pd.DataFrame()
        df_output["index"] = self.price_data["index"]
        df = self.price_data
        for ticker_i in stock_list:
            for ticker_j in stock_list:
                if (
                    ticker_j != ticker_i
                    and ticker_j + ticker_i not in run_list
                    and ticker_i + ticker_j not in run_list
                ):
                    df_output[ticker_j + "_" + ticker_i] = (
                        df[ticker_j].rolling(rolling_window).corr(df[ticker_i])
                    )
                    run_list.append(ticker_j + ticker_i)
                    run_list.append(ticker_i + ticker_j)
        df_output[f"{rolling_window}_corr"] = df_output.drop("index", axis=1).mean(
            axis=1
        )
        self.rolling_window = rolling_window
        self.correlations = df_output
        return df_output

    def plot_cross_correlations(self):
        self.correlations["std"] = self.correlations[
            f"{self.rolling_window}_corr"
        ].std()
        self.correlations["mean"] = self.correlations[
            f"{self.rolling_window}_corr"
        ].mean()
        self.correlations["std_1"] = (
            self.correlations["mean"] + self.correlations["std"]
        )
        self.correlations["std_2"] = (
            self.correlations["mean"] + 2 * self.correlations["std"]
        )
        self.correlations["std_minus_1"] = (
            self.correlations["mean"] - self.correlations["std"]
        )
        self.correlations["std_minus_2"] = (
            self.correlations["mean"] - 2 * self.correlations["std"]
        )
        self.correlations.plot(
            ltcode=f"Cross Correlations for {self.stock_list}",
            x="index",
            y=[
                f"{self.rolling_window}_corr",
                "std_1",
                "std_2",
                "std_minus_1",
                "std_minus_2",
                "mean",
            ],
        ).legend(loc="lower left")

    def get_r2_data(self, betas_list, ticker):
        list_df = []
        for item in betas_list:
            df = self.get_price_data(item)
            list_df.append(df)

        df_long = pd.concat(list_df, axis=0).reset_index()
        df_wide = df_long.pivot_table(
            index=["index"], columns="ticker", values="pct_change"
        )

        df_sq = self.get_price_data(ticker).reset_index()

        price_data = pd.merge(
            df_sq, df_wide, how="inner", left_on="index", right_on="index"
        )

        beta_lens = [60, 90, 252]  # window for betas
        dict_lens = {}
        for beta_len in beta_lens:
            betas = {}
            ns = {}
            r2s = {}
            for col in price_data.columns:
                if col in betas_list:
                    df = price_data[["pct_change", col]].dropna()
                    beta, r2, n = self.get_betas(
                        df["pct_change"], df[col], beta_len, col
                    )
                    betas[col] = beta
                    ns[col] = n
                    r2s[col] = r2
            dict_lens[beta_len] = [betas, ns, r2s]

        df1 = pd.DataFrame.from_dict(dict_lens[60][2], orient="index").reset_index()
        df1.columns = ["TICKER", "R2_60"]
        df2 = pd.DataFrame.from_dict(dict_lens[90][2], orient="index").reset_index()
        df2.columns = ["TICKER", "R2_90"]
        df3 = pd.DataFrame.from_dict(dict_lens[252][2], orient="index").reset_index()
        df3.columns = ["TICKER", "R2_252"]
        df = pd.merge(df1, df2)
        df = pd.merge(df, df3)
        df.sort_values("R2_60")
        self.r2_data = df

    def plot_r2_data(self, ticker):
        self.r2_data.set_index("TICKER")[["R2_60", "R2_90", "R2_252"]].plot(
            ltcode=f"{ticker} Relative R2", kind="bar"
        )

    def get_beta_data(self, stock_list):
        self.stock_list = stock_list
        list_df = []
        for item in stock_list:
            df = self.get_price_data(item)
            list_df.append(df)
        self.df_long = pd.concat(list_df, axis=0).reset_index()
        self.df_wide = self.df_long.pivot_table(
            index=["index"], columns="ticker", values="pct_change"
        )

        self.spy = self.get_price_data("SPY").reset_index()
        self.price_data = pd.merge(
            self.spy, self.df_wide, how="inner", left_on="index", right_on="index"
        )
        betas = {}
        ns = {}
        r2s = {}
        for col in self.price_data.columns:
            if col in self.stock_list:
                df = self.price_data[["pct_change", col]].dropna()
                beta, r2, n = self.get_betas(df["pct_change"], df[col])
                betas[col] = beta
                ns[col] = n
                r2s[col] = r2

        self.betas_dict = {
            k: v for k, v in sorted(betas.items(), key=lambda item: item[1])
        }
        self.price_data_stocks = self.price_data.drop(
            columns=["adjclose", "pct_change", "log_return", "ticker"]
        )
        self.portfolio_return = self.get_portfolio_return(
            self.price_data_stocks, self.stock_list, n=252
        )
        self.spy_return = self.get_portfolio_return(
            self.price_data[["pct_change"]], "pct_change", n=252
        )
        self.spy_return = self.spy_return.drop(
            columns=["pct_change", "pct_change_cmltv_ret"]
        )
        self.spy_return.columns = ["SPY_return"]

    def plot_relative_return(self):
        self.df_plot = pd.concat([self.portfolio_return, self.spy_return], axis=1)
        self.df_plot.set_index("index")[
            ["portfolio_return", "SPY_return", "SQ_cmltv_ret"]
        ].plot(ltcode="Cumulative Return")

    def get_price_data(self, ticker):
        price_data = si.get_data(ticker, start_date="01/01/2009")
        df = pd.DataFrame(price_data)
        df = df[["adjclose"]]
        df["pct_change"] = df.adjclose.pct_change()
        df["log_return"] = np.log(1 + df["pct_change"].astype(float))
        df["ticker"] = ticker
        return df

    def get_portfolio_return(self, df, list_stocks, n):
        df = df.iloc[-n:,].copy()
        for col in df.columns:
            if col in list_stocks:
                df[col + "_cmltv_ret"] = np.exp(np.log1p(df[col]).cumsum()) - 1
        list_cols = []
        for col in df.columns:
            if "cmltv" in col and "SPY" not in col:
                list_cols.append(col)
        df["portfolio_return"] = df[list_cols].mean(axis=1)
        return df

    def get_betas(self, x, y, n=0, col=""):
        if n > 0:
            x = x.iloc[-n:,]
            y = y.iloc[-n:,]
        res = sm.OLS(y, x).fit()
        ticker = col.split("_")[0]
        beta = res.params[0]
        r2 = res.rsquared
        n = len(x)
        return [beta, r2, n]
