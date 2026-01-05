from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.db.sqlite import SQLiteDB
import os
load_dotenv()

db = SQLiteDB
