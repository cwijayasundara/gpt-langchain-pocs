from sqlalchemy import MetaData

metadata_obj = MetaData()

from sqlalchemy import Column, Integer, String, Table, Date, Float
from sqlalchemy import create_engine
from datetime import datetime
from sqlalchemy import insert

customers = Table(
    "customers",
    metadata_obj,
    Column("customer_id", Integer, primary_key=True),
    Column("customer_name", String(50), nullable=False),
    Column("customer_address", String(200), nullable=False),
    Column("date_joined", Date, nullable=False),
)
accounts = Table(
    "accounts",
    metadata_obj,
    Column("account_id", Integer, primary_key=True),
    Column("customer_id", Integer),
    Column("account_balance", Float, nullable=False),
    Column("date_opened", Date, nullable=False),
)

engine = create_engine("sqlite:///:memory:")

observations_1 = [
    [1, 'Tom Johns', '14, W close, Watford, WD17 5PP', datetime(1974, 1, 1)],
    [2, 'James Arther', '234, London, LD34 P99', datetime(1950, 1, 2)]
]
observations_2 = [
    [1202034, 1, 5000, datetime(2010, 1, 2)],
    [3456782, 2, 2000, datetime(2020, 1, 2)]
]


def insert_obs(obs):
    stmt = insert(customers).values(
        customer_id=obs[0],
        customer_name=obs[1],
        customer_address=obs[2],
        date_joined=obs[3]
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def insert_obs_acct(obs):
    stmt = insert(accounts).values(
        account_id=obs[0],
        customer_id=obs[1],
        account_balance=obs[2],
        date_opened=obs[3]
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def set_up_db():
    metadata_obj.create_all(engine)
    for obs in observations_1:
        insert_obs(obs)
    for obs in observations_2:
        insert_obs_acct(obs)
