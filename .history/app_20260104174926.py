from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.db.sqlite import SqliteDb
import os
from agno.tools.csv_toolkit import CsvTools
from pathlib import Path
from agno.tools.file import FileTools
from agno.tools.pandas import PandasTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools
from agno.tools.visualization import VisualizationTools
load_dotenv()

db = SqliteDb(db_file = "memory.db" , session_table = "session_table")

#======================DataDir======================
base_dir = Path(__file__).parent
# path for the data 
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
           FileTools(base_dir=base_dir)]
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
    tools=[FileTools(base_dir=base_dir)]
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
    tools=[PandasTools(), FileTools(base_dir=base_dir)],
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
    tools=[PandasTools(), FileTools(base_dir=base_dir), VisualizationTools("plots")],
    instructions=["You are an expert in creating plots using matplotlib",
                  "you have access to the files tool to list down the data files in the project folder",
                  "you also have access to pandas tools that can run pandas specific code that will be used as an input to create charts",
                  "you can create charts like bar plots, pie charts, line plotsm, histograms and scatter plots",
                  "use bar plots and histograms to plot univariate charts on categorical and numerical columns respectively",
                  "when studying the relationship between two numerical columns, you can use a scatter plot",
                  "always make sure to input the correct data for the respective chart type"]
)
# Create Coding Agent
coding_agent = Agent(
    id="coding-agent",
    name="Coding Agent",
    db=db,
    add_history_to_context=True,
    role="Python Coding Assistant",
    num_history_runs=5,
    read_chat_history=True,
    model=model,
    tools=[DuckDuckGoTools(), PythonTools(base_dir=base_dir), ShellTools(base_dir=base_dir)],
    instructions=["you are an expert coding agent proficient in writing python code",
                  "your main task is to write code specific to Machine Learning",
                  "you may be using numpy pandas sklearn scipy etc. in your code",
                  "you have access to tool that can list files in the directory and can also read them",
                  "the python tool also lets you create python files and write them, you can also run them to get the desired output",
                  "you will be used to write python code for ML and Data Science specific tasks such as data cleaning, feature engineering, model training and model evaluation",
                  "you also have web search capability in case you need to access the latest documentation from the web",
                  "make sure to get the code reviewed by the user and only write it into a file when the user accepts it",
                  "if you want to add packages use the shell tool and use the command `uv add <package_name>`",
                  "do not use the shell tool to execute any other command than the one mentioned above"]
)

