from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.db.sqlite import SQLiteDB
import os
from agno.tools.csv_toolkit import CsvTools
from pathlib import Path

load_dotenv()

db = SQLiteDB(db_file = "memory.db" , session_table = "session_table")

#======================DataDir======================

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)
# =======================Models======================

model = Gemini(
    model_name = "gemini-2.5-pro",)


# =======================Agents======================

# Create data loader agent
data_loader_agent = Agent(
    id = "data_loader_agent",
    name = "data_loader_agent",
    model  = model
    add_history_to_context=True,
    num_history_messages=3,
    instructions=["You are an Expert in loading csv files from the project folder",
                  "YOu hae the capabilityto list Down csv files and to read them",
                  "make sure to not read more than 20 to 30 rows inside the csv file",
                  ],
    tools = [CsvTools(enable_query_csv_file = False)]

)