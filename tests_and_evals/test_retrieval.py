# test_retrieval.py
import os, sys
from supabase import create_client, Client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from .. import config, llm_select
from config import SUPABASE_VS_URL, SUPABASE_VS_KEY
from llm_select import embeddings, embedding_model_name, TABLE_NAME

# --- Hardcoded user issue ---
user_issue = "During climb the autopilot disconnects unexpectedly."
aircraft_type = "MU2"

# --- 1. Embed the issue ---
vec = embeddings.embed_query(user_issue)
dim = len(vec)
print(f"Embedded query into {dim}-dim vector using {embedding_model_name}")

# --- 2. Connect to Supabase client ---
supabase_vs: Client = create_client(SUPABASE_VS_URL, SUPABASE_VS_KEY)

# --- 3. Run similarity search ---
# Assumes you created a Postgres function (RPC) for vector similarity search,
# e.g. `match_manuals(query_embedding vector, match_count int)`
# --- 3. Run similarity search ---
# Build a pgvector literal: '[v1,v2,...]'
vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"  # compact, no spaces

sql = f"""
    SELECT
    id, content, page_number, section_title, aircraft_model, document_id, system_category,
    embedding <#> '{vec}'::vector AS neg_ip_distance,
    -(embedding <#> '{vec}'::vector)       AS cosine_sim
    FROM {TABLE_NAME}
    WHERE aircraft_model = 'MU2'
    ORDER BY neg_ip_distance ASC
    LIMIT 20;
"""

response = supabase_vs.rpc("execute_sql", {"sql": sql}).execute()
rows = getattr(response, "data", response)

print("\n=== Retrieved Context ===")
for r in rows:
    # you selected AS distance, so print that
    dist = r.get("neg_ip_distance")
    sim  = r.get("cosine_sim", (-dist if dist is not None else 0.0))
    print(f"[cos {sim:.3f}] {r.get('content')}")

