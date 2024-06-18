#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:36:06 2024

@author: christopherdignam
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from tools.vector import kg_qa


# Include the LLM from a previous lesson
from llm import llm

# How many recent messages to keep in conversation memory ?
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

# List of tools available to the Agent
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
    ),
    Tool.from_function(
        name="Vector Search Index",  # (1)
        description="Provides information about EU AI Act terms using Vector Search",  # (2)
        func=kg_qa,  # (3)
        return_direct=True
    )
]

# Define the agent prompt
# https://smith.langchain.com/hub/
# https://smith.langchain.com/hub/hwchase17/react-chat?organizationId=d9a804f5-9c91-5073-8980-3d7112f1cbd3
# agent_prompt = hub.pull("hwchase17/react-chat")
agent_prompt = PromptTemplate.from_template("""
You are a expert on the EU AI Act legislation.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to the EU AI Act legislation, terms found in the EU AI Act and AI in general.
Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=False  # try to see what happens ! Delete if not useful.
    )


def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    print("The prompt is: %r" % prompt)  # for debug
    try:
        response = agent_executor.invoke({"input": prompt})
        # response = agent_executor.invoke({"input": "How are you?"})
    except Exception as e:
        print(f"There was an error generating the response: {e}")
    print("Here is the response: %r" % response)
    # print("The response is as follows: %r" % response['output'])  # for debug

    return response['output']
