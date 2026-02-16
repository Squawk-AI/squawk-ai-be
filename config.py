
import os
from supabase import Client, create_client



from dotenv import load_dotenv
load_dotenv()

# --- CONFIG: Choose model provider here for embeddings (embedding query and retrieving context) ---
EMBEDDING_MODEL_PROVIDER = "openai"  # "openai" or "huggingface"


#Vector store database for manuals
SUPABASE_VS_URL = os.environ.get("SUPABASE_VS_URL")
SUPABASE_VS_KEY = os.environ.get("SUPABASE_VS_KEY")
supabase_VS: Client = create_client(SUPABASE_VS_URL, SUPABASE_VS_KEY)

#Overview database for aircraft overviews
SUPABASE_OV_URL = os.environ.get("SUPABASE_OV_URL")
SUPABASE_OV_KEY = os.environ.get("SUPABASE_OV_KEY")
supabase_OV: Client = create_client(SUPABASE_OV_URL, SUPABASE_OV_KEY)

#Chat database for conversation logging
SUPABASE_CHAT_URL = os.environ.get("SUPABASE_CHAT_URL")
SUPABASE_CHAT_KEY = os.environ.get("SUPABASE_CHAT_KEY")
supabase_CHAT: Client = create_client(SUPABASE_CHAT_URL, SUPABASE_CHAT_KEY)

