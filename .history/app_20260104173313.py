from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.db.sqlite import SqliteDb
import os
from agno.tools.csv_toolkit import CsvTools
from pathlib import Path
from agno.tools.file import FileTools
load_dotenv()

db = SqliteDb(db_file = "memory.db" , session_table = "session_table")

#======================DataDir======================

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)
data_path = Path(__file__).parent / "data" / "car_details.csv"
# =======================Models======================

model = Gemini(
    id="gemini-2.5-flash"
)

# =======================Agents======================

# Create data loader agent
data_loader_agent = Agent(
    id = "data_loader_agent",
    name = "data_loader_agent",
    model  = model,
    add_history_to_context=True,
    db=db,
    num_history_messages=3,
    instructions=["You are an Expert in loading csv files from the project folder",
                  "YOu hae the capabilityto list Down csv files and to read them",
                  "make sure to not read more than 20 to 30 rows inside the csv file",
                  ],
    tools=[CsvTools(enable_query_csv_file=False, csvs=[data_path]),
           FileTools(base_dir=data_dir)]
)

file_manager_agent = Agent(
    id="file-manager-agent",
    name="File Manager Agent",
    model=model,
    role="Manages Filesystem",
    instructions=["You are an expert File Management Agent",
                  "Your task is to list down files when asked to do it",
                  "you can also read and write files",
                  "make sure to never read csv files, you can only list them"],
    tools=[FileTools(base_dir=data_dir)]
)

if __name__ == "__main__":
    data_loader_agent.cli_app()