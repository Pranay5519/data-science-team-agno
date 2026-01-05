from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.team import Team
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
#Shell agent
shell_agent = Agent(
    id="shell-agent",
    name="Shell Agent",
    model=model,
    db=db,
    add_history_to_context=True,
    tools=[ShellTools(base_dir=base_dir)],
    role="Shell Commands Executor",
    instructions=["You have the capability to run shell commands",
                  "use this capability with extreme caution",
                  "only use the shell tool capability if unable to execute the python file",
                  "use the command `uv run <python_file.py>`",
                  "do not use the shell tool for commands that can make changes in the project structure",
                  "do not delete any file",
                  "just use it to read the project structure, files or to run files"]
)
# ===================TEAM===============
# data Science Team
data_science_team = Team(
    members=[data_loader_agent,
             file_manager_agent,
             data_understanding_agent,
             visualization_agent,
             coding_agent,
             shell_agent],
    id="data-science-team",
    name="Data Science Team",
    role="Team Leader / Project Manager",
    model=model,
    instructions=["You are an expert Data Scientist and a team leader",
                  "you manage team members which are good at coding, using pandas, generate visualizations, using matplotlib, executing shell commands, loading data files and also managaing the project filesystem",
                  "delegate tasks to your members according to their roles and capabilities",
                  "your task is to assist the user through a machine learning pipeline having steps such as data loading, data understanding, data cleaning, plotting of charts, feature engineering, model training and model evaluation",
                  "whenever you generate code, first get it reviewed by the user before saving and executing it",
                  "try to assist the user at all times and provide steps for each  stage so that the user knows what to do",
                  "you have access to a session state where you can add session wise memory",
                  "always try to add important stuff such as data path, path to models and path to src files",
                  "whenever the user asks you to remember anything, just add it to the session state so that it can be retrieved later",
                  "always try to assist the user with ideas and always think like a professional data scientist",
                  "if you get any errors try to have an approach to debug and solve issues"],
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    read_chat_history=True,
    session_state={},
    add_session_state_to_context=True,
    add_member_tools_to_context=True,
    enable_agentic_state=True
)
# ==================== Agent OS =======================================
agent_os = AgentOS(
    id='agent-os',
    name="Data Science Team",
    description="This team of agents helps you throughout your data science and ML pipeline journey",
    teams=[data_science_team]
)

if __name__ == "__main__" :
    visualization_agent.cli_app()
    