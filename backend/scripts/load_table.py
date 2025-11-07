import pandas as pd
import sqlite3

df = pd.read_csv("../sql/citations_paracetamol.csv")

conn = sqlite3.connect("/home/srini/PycharmProjects/medicine-genai-langgraph/data/medical.db")  # path to your existing DB
df.to_sql("citations", conn, if_exists="append", index=False)
conn.close()
