from marshmallow import Schema, fields, post_load
import datetime as dt
from dataclasses import dataclass
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd

import financedatabase as fd

import yahoo_fin.stock_info as si

from src.data_objects import IOData, Portfolio

import os
import ftplib


DATA_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = ROOT_DIR.replace("/src", "")
DATA_DIR = f"{ROOT_DIR}/data"

FILE_TYPE = "csv"  # or parquet


@dataclass
class DataSource:
    focus_list: list
    is_global: str
    name: str
    user_name: str
    password: str
    url: str
    hostname: str
    api_key: str
    date_field: str
    data_field: str
    category_field: str
    kpi_name_field: str
    add_features_true_false: str = "True"
    created_at = dt.datetime.now()

    def financedb(self):
        """
        set up ticker reference - security master
        """

        equities = fd.Equities()
        df = equities.select(country="United States")
        return df

    def insync(self, ticker):
        # Connect FTP Server
        ftp_server = ftplib.FTP(self.hostname, self.user_name, self.password)

        # force UTF-8 encoding
        ftp_server.encoding = "utf-8"
        dir_list = ftp_server.nlst()
        ftp_server.cwd(ticker)
        dir_list = ftp_server.nlst()
        filename = f"DATA/{ticker}/{ticker}_actuals.csv"
        for file in dir_list:
            if "Actuals" in file:
                source_file = file

        with open(filename, "wb") as file:
            ftp_server.retrbinary(f"RETR {source_file}", file.write)
        filename = f"DATA/{ticker}/{ticker}_consensus.csv"
        return None

    def alphavantage(self, ticker):
        # 5 calls per minute
        import time

        time.sleep(10)
        endpoint = self.url
        response = requests.get(f"{endpoint}&symbol={ticker}&apikey={self.api_key}")
        df = response.json()
        try:
            df = pd.DataFrame(df["quarterlyEarnings"])
            df["ticker"] = ticker
        except:
            print(f"AlphaVantage error for {ticker}")
            print(df)
            return None
        return df

    def yfinance(self, ticker):
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
        return df

    def get_data(self):
        self.io = IOData(data_cache=FILE_TYPE, data_source=self.name)

        if self.is_global == "True":
            try:
                func = getattr(self, self.name)
                df = func()
                self.save_data(df, self.name)
                return
            except:
                return

        for ticker in self.focus_list:
            func = getattr(self, self.name)
            df = func(ticker)

            if self.category_field != "":
                categories = df[self.category_field].unique()
                for item in categories:
                    df_out = df[df[self.category_field] == item]
                    if self.add_features_true_false == "True":
                        df_out = self.add_features(df_out)
                    self.save_data(df_out, ticker, item)
            else:
                if self.add_features_true_false == "True":
                    df = self.add_features(df)
                if df is not None:
                    self.save_data(df, ticker)

    def save_data(self, df, ticker, item=""):
        self.io.save(df, ticker, item)
        return

    def add_features(self, df):
        df = df.copy()
        df[self.date_field] = df[self.date_field].astype(str)

        df[self.date_field + "_ym"] = (
            df[self.date_field].str.split("-").str[0]
            + "-"
            + df[self.date_field].str.split("-").str[1]
        )

        df[self.date_field] = pd.to_datetime(df[self.date_field])

        return df

    def __repr__(self):
        return self.name


class DataSourceSchema(Schema):
    focus_list = fields.List(fields.Str())
    is_global = fields.Str()
    name = fields.Str()
    user_name = fields.Str()
    password = fields.Str()
    hostname = fields.Str()
    url = fields.Str()
    api_key = fields.Str()
    date_field = fields.Str()
    data_field = fields.Str()
    category_field = fields.Str()
    kpi_name_field = fields.Str()
    add_features_true_false = fields.Str()
    created_at = fields.DateTime()

    @post_load
    def make_datasource(self, data, **kwargs):
        return DataSource(**data)
