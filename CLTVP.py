# !pip install lifetimes
# !pip install sqlalchemy
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
# !pip3 install --upgrade mysql-connector-python
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# PREPARING DATASET FOR OUTLIERS, MISSING OBSERVATIONS..

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df_ = pd.read_excel("../online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.dropna(inplace = True)

# DELETING RETURNS FROM THE DATASET
df = df[~df["Invoice"].str.contains("C", na=False)]

df = df[df["Quantity"] > 0 ]

# WE WILL WORK ON UNITED KINGDOM
df = df[df["Country"] == "United Kingdom"]

# SUPPRESSING OUTLINES WITH LOWER AND UPPER LIMIT VALUES
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = pd.datetime(2011,12,11)

# WE NEED TO GET RECENCY, FREQUENCY, MONETARY AND T(AGE OF CUSTOMER) FOR EVERY CUSTOMER; SO, WE WILL DO GROUPBY ON CUSTOMER ID.
cltv_p = df.groupby('Customer ID').agg({'InvoiceDate': [lambda x: (x.max() - x.min()).days,
                                                lambda x: (today_date - x.min()).days],
                                                'Invoice': lambda x: x.nunique(),
                                                'TotalPrice': lambda x: x.sum()})

cltv_p.columns = cltv_p.columns.droplevel(0)

cltv_p.columns = ["Recency", "T", "Frequency", "Monetary"]

# WE HAVE REGULATED MONETARY TO REFER TO THE AVERAGE EARNINGS PER PURCHASE
cltv_p["Monetary"] = cltv_p["Monetary"] / cltv_p["Frequency"]

cltv_p = cltv_p[cltv_p["Monetary"] > 0]

# WEEKLY EXPRESSION OF RECENCY AND T
cltv_p["Recency"] = cltv_p["Recency"] / 7
cltv_p["T"] = cltv_p["T"] / 7

# WE SELECT CUSTOMERS WHO HAVE SHOPPED AT LEAST TWICE
cltv_p = cltv_p[(cltv_p['Frequency'] > 1)]

# BG/NBD MODEL FOR 6-MONTH
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_p['Frequency'],
        cltv_p['Recency'],
        cltv_p['T'])

cltv_p["expected_purchases_for_6_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_p["Frequency"],
                                                        cltv_p["Recency"],
                                                        cltv_p["T"])

# GAMMA GAMMA MODEL
ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_p["Frequency"],
        cltv_p["Monetary"])
cltv_p["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_p["Frequency"], cltv_p["Monetary"])

# LOOK AT THE 5 MOST PROFITABLE CUSTOMERS
ggf.conditional_expected_average_profit(cltv_p["Frequency"], cltv_p["Monetary"]).sort_values(ascending=False).head()

# CLTV PREDICTION FOR 6-MONTH
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_p["Frequency"],
                                   cltv_p["Recency"],
                                   cltv_p["T"],
                                   cltv_p["Monetary"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

# LET'S MERGE "cltv" AND "cltv_p" DATAFRAMES.
cltv = cltv.reset_index()
cltv_final = cltv_p.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False)

# THROUGH THE 6-MONTH RESULTS, WE DIVERSE CUSTOMERS IN 4 SEGMENTS ACCORDING TO THEIR CLV VALUES.
cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels = ["D", "C", "B", "A"])
cltv_final.sort_values(by="clv", ascending=False)

# IF WE WANT TO MAKE SEGMENTS SPECIAL INSPECTION, WE CAN USE THE FOLLOWING FUNCTION IN A FOR LOOP.
seg_list = ["A", "B", "C", "D"]
agg_list = ["mean","min","max","sum"]
def check_clv(dataframe, variable_name1, variable_name2, seg_name, agg_type):
    print("##########-"+str(seg_name)+"-##########")
    print(dataframe[dataframe[variable_name1] == seg_name].agg({variable_name2 : agg_type}))

for seg in seg_list:
    check_clv(cltv_final, "Segment", "clv", seg, agg_list)

# OR WE CAN USE GROUPBY
cltv_final.groupby("Segment").agg(["mean", "count", "sum"])

# THINK THAT WE HAVE A CONNECTER WHICH IS CALLED CONN. WE WILL SEND OUR DATASET TO DATABASE WITH USING THIS CONNECTER.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

conn = create_engine(connstr.format(**creds))

cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)

cltv_final.to_sql(name='BAHAR_ZERENTURK', con=conn, if_exists='replace', index=False)
