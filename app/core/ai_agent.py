from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

from app.config.settings import settings

def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):
    try:
        llm = ChatGroq(model=llm_id)
        tools = [TavilySearch(max_results=2)] if allow_search else []
        
        # create_react_agent uses 'prompt' parameter, not 'state_modifier'
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt
        )

        state = {"messages": query}
        response = agent.invoke(state)
        messages = response.get("messages")
        ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
        
        if not ai_messages:
            raise ValueError("No AI response messages found in agent output")
            
        return ai_messages[-1]
    except Exception as e:
        raise Exception(f"AI Agent Error - Model: {llm_id}, Error: {str(e)}")