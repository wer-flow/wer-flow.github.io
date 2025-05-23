import operator
import unicodedata
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    List,
    Sequence,
    cast,
)

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import CustomStreamEvent, StandardStreamEvent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, SecretStr
from typing_extensions import TypedDict


async def _compose_answer(self, state: BaseAgentState) -> dict:
    """
    Generates an AI response based on the current query and context using a streaming model.

    Combines the query with any available document context, builds a prompt, and then
    chains the prompt with the LLM model to obtain the response.

    Args:
        state (BaseAgentState): The current conversation state containing the query and documents.

    Returns:
        dict: A dictionary with a key "messages" wrapping the generated response.
    """
    query = state['user_text']
    await adispatch_custom_event(FoundationModelCalledEvent.NAME, {"knowledge_list": state["knowledge_items"], "system_prompt": ""})

    assert self._answer_generator_system_prompt_template is not None
    assert self._answer_generator_user_prompt_template is not None

    # Build the prompt using system and human message templates.
    prompt = ChatPromptTemplate.from_messages([
        ("system", self._answer_generator_system_prompt_template),
        ("human", self._answer_generator_user_prompt_template)
    ])

    # Instantiate the LLM with the given API key and temperature.
    model = ChatOpenAI(
        api_key=self._model_api_key,
        model="o4-mini",
        reasoning_effort="high",
        temperature=None,
        seed=41,
        stream_usage=True,
    )

    # Chain the prompt with the model.
    chain = prompt | model

    config = RunnableConfig(tags=[self.TAG_AGENT])
    response = await chain.ainvoke({
        "query": query,
        "context": self._format_knowledge_items_to_str(state["knowledge_items"]),
        "datetime": current_datetime()
    }, config=config)

    return {"messages": [response], "completion": unicodedata.normalize('NFKC', str(response.content))}