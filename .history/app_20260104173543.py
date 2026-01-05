from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.db.sqlite import SqliteDb
import os
from agno.tools.csv_toolkit import CsvTools
from pathlib import Path
from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from agno.tools.pandas import PandasTools
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
# File Management Agent
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
#Data Understanding Agent
data_understanding_agent = Agent(
    id="data-understanding-agent",
    name="Data Understanding Agent",
    model=model,
    role="Data understanding and Exploration assistant",
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    instructions=["You are an expert in handling pandas operations on a df",
                  "you can create df and also perform operations on it",
                  "the operations you perform on a df are head(), tail(), info(), describe() for numerical columns and you can also calculate the value_counts() for categorical columns",
                  "make sure to list down the numerical and categorical columns in the df",
                  "you can also check the shape of the df using the .shape attribute",
                  "you also have access to tools which can search for data files"
                  ],
    tools=[PandasTools(), FileTools(base_dir=data_dir)],
)
# create a visualization agent
visualization_agent = Agent(
    id="viz-agent",
    name="Visualization Agent",
    model=model,
    role="Plotting and Visualization Assistant",
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    tools=[PandasTools(), FileTools(base_dir=data_dir), VisualizationTools("plots")],
    instructions=["You are an expert in creating plots using matplotlib",
                  "you have access to the files tool to list down the data files in the project folder",
                  "you also have access to pandas tools that can run pandas specific code that will be used as an input to create charts",
                  "you can create charts like bar plots, pie charts, line plotsm, histograms and scatter plots",
                  "use bar plots and histograms to plot univariate charts on categorical and numerical columns respectively",
                  "when studying the relationship between two numerical columns, you can use a scatter plot",
                  "always make sure to input the correct data for the respective chart type"]
)

if __name__ == "__main__":
    data_loader_agent.cli_app()