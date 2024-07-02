import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Travel Checklist Generator")
st.markdown("Welcome to Travel Checklist Generator, your personalized travel packing assistant! Simply input your destination and trip duration, and get a customized packing list tailored to your needs.")
st.markdown("            1) Name of your travel destination.")
st.markdown("            2) Mention the trip duration.")
st.markdown("            3) Provide additional information if any like planned activities, accomodation type and others.")
input = st.text_input(" Please enter the above details:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role=" Expert TRAVEL PLANNER ",
        prompt_persona=f"Your task is to DEVELOP a COMPREHENSIVE and CUSTOMIZED travel checklist for a user, based on the SPECIFIC INFORMATION they provide about their destination, trip duration, activities they plan to engage in, accommodation type, and any other relevant details.")
    prompt = f"""
You are an Expert TRAVEL PLANNER. Your task is to DEVELOP a COMPREHENSIVE and CUSTOMIZED travel checklist for a user, based on the SPECIFIC INFORMATION they provide about their destination, trip duration, activities they plan to engage in, accommodation type, and any other relevant details.

To ensure a THOROUGH and USEFUL checklist, follow these steps:

1. ANALYZE the information provided by  the user about their DESTINATION and other optional information like including climate, cultural norms, local regulations and list of ACTIVITIES that might affect what they need to pack or prepare for. IDENTIFY the DURATION of the trip to determine the quantity of items they will need to bring along. IDENTIFY the information on the TYPE of ACCOMMODATION (e.g., hotel, hostel, camping) as different lodgings may necessitate different essentials.

2. CREATE a checklist that includes all ESSENTIALS such as travel documents, clothing suitable for the destination's weather and planned activities, toiletries, electronics, health and safety items, and any additional gear specific to their needs.

3. ORGANIZE this checklist into CATEGORIES (e.g., documents, clothing) for EASY REFERENCE.

You MUST ensure that every item on this list is TAILORED to the user's unique travel plans.

"""

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Generate"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)