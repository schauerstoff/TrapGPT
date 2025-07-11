
from pathlib import Path
import yaml
from graphrag.config.load_config import load_config
import tiktoken
from graphrag.config.enums import ModelType
from graphrag.language_model.manager import ModelManager
import pandas as pd
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.structured_search.basic_search.search import BasicSearch
from graphrag.query.structured_search.basic_search.basic_context import BasicSearchContext

# Settings
LLM = "gpt-4o-mini" # gpt-4o, gpt-4o-mini, gpt-4.1
EMBEDDING = "embedding-small" # embedding-small

RESPONSE_TYPE = "Answer in German. Give a concise answer in plain text without formatting! Do not mention sources, or data reports in your answer."
COMMUNITY_LEVEL = 2


# this could be cleaner but there is no good wrapper for this...

INPUT_DIR = "trapgpt/graphrag/output"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_REPORT_TABLE = "community_reports"
ENTITY_TABLE = "entities"
COMMUNITY_TABLE = "communities"
RELATIONSHIP_TABLE = "relationships"
COVARIATE_TABLE = "covariates"
TEXT_UNIT_TABLE = "text_units"

PROJECT_DIRECTORY = "trapgpt/graphrag" 
settings = yaml.safe_load(open(f"{PROJECT_DIRECTORY}/settings.yaml")) 
graphrag_config = load_config(Path(PROJECT_DIRECTORY))

chat_config = graphrag_config.models["default_chat_model"] 
embedding_config = graphrag_config.models["default_embedding_model"] 

def get_chat_model():
    chat_model = ModelManager().get_or_create_chat_model(
    name="chat_model",  
    model_type=ModelType.OpenAIChat,
    config=chat_config,
    )
    return chat_model

def get_text_embedder():
    text_embedder = ModelManager().get_or_create_embedding_model(
    name="embedding_model",
    model_type=ModelType.OpenAIEmbedding,
    config=embedding_config,
    )
    return text_embedder

token_encoder = tiktoken.encoding_for_model(chat_config.model)

description_embedding_store = LanceDBVectorStore(
    collection_name="default-entity-description",
)
description_embedding_store.connect(db_uri=LANCEDB_URI)

chunk_embedding_store = LanceDBVectorStore(
    collection_name="default-text_unit-text",
)
chunk_embedding_store.connect(db_uri=LANCEDB_URI)

entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")

relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)

report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")

text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)

## global
community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")

communities = read_indexer_communities(community_df, report_df)
reports = read_indexer_reports(report_df, community_df, community_level=None, dynamic_community_selection=True)
entities = read_indexer_entities(entity_df, community_df, community_level=COMMUNITY_LEVEL)

def create_vector_search_engine():
    chat_model = get_chat_model()   
    text_embedder = get_text_embedder()

    basic_context_builder = BasicSearchContext(
        text_embedder=text_embedder,
        text_unit_embeddings=chunk_embedding_store,
        text_units=text_units,
        token_encoder=token_encoder,
        embedding_vectorstore_key=EntityVectorStoreKey.ID 
    )

    model_params = {
    "max_tokens": 2_000,
    "temperature": 0.0,
}

    basic_context_params = {
    "k":10
    }

    basic_search_engine = BasicSearch(
        model=chat_model,
        context_builder=basic_context_builder,
        token_encoder=token_encoder,
        model_params=model_params,
        context_builder_params=basic_context_params,
        response_type=RESPONSE_TYPE,  
    )

    ModelManager.get_instance().remove_chat(name="chat_model")
    ModelManager.get_instance().remove_chat(name="embedding_model")
    return basic_search_engine
     
def create_local_search_engine():
    chat_model = get_chat_model()   
    text_embedder = get_text_embedder()

    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=None,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,  
        "max_tokens": 8_000,  
    }

    model_params = {
        "max_tokens": 2_000,  
        "temperature": 0.0,
    }

    local_search_engine = LocalSearch(
        model=chat_model,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        model_params=model_params,
        context_builder_params=local_context_params,
        response_type=RESPONSE_TYPE, 
    )

    ModelManager.get_instance().remove_chat(name="chat_model")
    ModelManager.get_instance().remove_chat(name="embedding_model")
    return local_search_engine

def create_global_search_engine():
    chat_model = get_chat_model()   
    text_embedder = get_text_embedder()

    global_context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,  
        token_encoder=token_encoder,
    )

    global_context_builder_params = {
        "use_community_summary": False, 
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,  
        "temperature": 0.0,
    }

    global_search_engine = GlobalSearch(
        model=chat_model,
        context_builder=global_context_builder,
        token_encoder=token_encoder,
        max_data_tokens=8_000,  
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False, 
        json_mode=True,  
        context_builder_params=global_context_builder_params,
        concurrent_coroutines=32,
        response_type=RESPONSE_TYPE,
    )

    ModelManager.get_instance().remove_chat(name="chat_model")
    ModelManager.get_instance().remove_chat(name="embedding_model")
    return global_search_engine

def get_token_and_cost(search_result):
        input_tokens = sum(search_result.prompt_tokens_categories.values())
        output_tokens = sum(search_result.output_tokens_categories.values())
        return input_tokens + output_tokens, calculate_costs(input_tokens,  output_tokens, LLM)

def calculate_costs(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = {
        "gpt-4o": {"input": 2.5, "output": 10},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4.1": {"input": 0.04, "output": 1.6},
        "embedding-small": {"input": 0.02, "output": 0.0},
    }

    if model not in pricing:
        raise ValueError(f"Unbekanntes Modell: {model}")

    p = pricing[model]
    dollar_cost = input_tokens * (p["input"] / 1_000_000) + output_tokens * (p["output"] / 1_000_000)
    euro_cost = dollar_cost * 0.86
    return euro_cost

async def vector_search(query):
        vector_search_engine = create_vector_search_engine()
        search_result = await vector_search_engine.search(query)
        answer = search_result.response
        token_usage, cost = get_token_and_cost(search_result)
        return answer, token_usage, cost
        return "vector", 100, 0.01

async def local_search(query):
        local_search_engine = create_local_search_engine()
        search_result = await local_search_engine.search(query)
        answer = search_result.response
        token_usage, cost = get_token_and_cost(search_result)
        return answer, token_usage, cost

async def global_search(query):
        global_search_engine = create_global_search_engine()
        search_result = await global_search_engine.search(query)
        answer = search_result.response
        token_usage, cost = get_token_and_cost(search_result)
        return answer, token_usage, cost

