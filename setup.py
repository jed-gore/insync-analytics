from src.data_objects import Portfolio
from src.data_sources import DataSourceSchema
import json

if __name__ == "__main__":
    FOCUS_LIST = "AAP,ADBE,AMZN,LMT,LOW,ORLY".split(",")

    datasources = {}

    with open("data_sources.json") as f:
        data = f.read()

    js = json.loads(data)

    for key in js.keys():
        print(f"ETL for {key}")
        datasource_data = js[key]
        datasource_data["focus_list"] = FOCUS_LIST
        schema = DataSourceSchema()
        result = schema.load(datasource_data)
        result.get_data()

    pm = Portfolio(FOCUS_LIST)

    pm.get_alphas()
    pm.get_fundamental_data()
    pm.get_rolling_ols()
