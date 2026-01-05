from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.db.sqlite import SQLiteDB
import os
load_dotenv()

db = SQLiteDB(db_file = "memory.db" , session_table = "session_table")

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
    instructions=[],
    

)