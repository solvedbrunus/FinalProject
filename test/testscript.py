import os
import re
import time
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pydantic import Field
import openai
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent  # Updated import
from langchain.prompts.chat import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.tools.base import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent
from langchain.tools import Tool
import pinecone
from typing import Any, Optional
import pygame
import io
from utils.audio_manager import AudioManager

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Load DataFrame
csv_file = 'dataset/healthcare_dataset.csv'
df = pd.read_csv(csv_file)

# Healthcare Data Frame Tool
class HealthcareDataFrameTool(BaseTool):
    name: str = Field(default="Healthcare Data Analysis Tool")
    description: str = Field(default="Tool for analyzing healthcare dataset using pandas operations")
    df: pd.DataFrame = Field(description="Healthcare dataset as a pandas DataFrame")

    def _run(self, query: str) -> Any:
        try:
            if "average" in query.lower() or "mean" in query.lower():
                if "billing" in query.lower():
                    return f"Average billing amount: ${self.df['Billing Amount'].mean():.2f}"
                elif "age" in query.lower():
                    return f"Average age: {self.df['Age'].mean():.1f} years"
            elif "count" in query.lower():
                if "patient" in query.lower():
                    return f"Total number of patients: {len(self.df)}"
                elif "condition" in query.lower():
                    return self.df['Medical Condition'].value_counts().to_string()
            elif "search" in query.lower():
                search_term = query.lower().split("search")[-1].strip()
                results = self.df[self.df['Name'].str.lower().str.contains(search_term, na=False)]
                return f"Found {len(results)} matching patients:\n{results[['Name', 'Age', 'Medical Condition']].to_string()}"
            return "Could not process the query. Please try rephrasing your question."
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

# Initialize LLM and memory
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

conversational_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create tool instance and agent
healthcare_tool = HealthcareDataFrameTool(df=df)
agent_test = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    tools=[healthcare_tool],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    max_iterations=10,
    allow_dangerous_code=True,
)

def query_healthcare_data(question: str) -> str:
    try:
        response = agent_test.run(question)
        return response
    except Exception as e:
        return f"Error processing question: {str(e)}"

# Initialize Embedding Model
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=pinecone_api_key)
spec = pinecone.ServerlessSpec(cloud="aws", region='us-east-1')

# Create Pinecone index
index_name = "healthcare-qa-pdfs"
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
index = pc.Index(index_name)

# Text processing functions
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            preprocessed_text = preprocess_text(text)
            chunks = chunk_text(preprocessed_text)
            texts.extend(chunks)
    return texts

def create_embeddings(texts):
    embeddings = embed.embed_documents(texts)
    return embeddings

# Process PDFs
directory_path = "healthcare_pdfs"
all_texts = []
all_embeddings = []
all_ids = []
for file_name in os.listdir(directory_path):
    if file_name.endswith('.pdf'):
        file_path = os.path.join(directory_path, file_name)
        print(f"Processing file: {file_path}")
        texts = process_pdf(file_path)
        if texts:
            print(f"Extracted {len(texts)} texts from {file_path}")
            embeddings = create_embeddings(texts)
            print(f"Created {len(embeddings)} embeddings for {file_path}")
            all_texts.extend(texts)
            all_embeddings.extend(embeddings)
            all_ids.extend([f"{file_path}_{i}" for i in range(len(embeddings))])
        else:
            print(f"No text extracted from {file_path}")

# Initialize Vector Store
text_field = "text"
vectorstore = Pinecone(index, embed.embed_query, text_field)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# System prompt and template
SYSTEM_PROMPT = """You are an advanced AI Healthcare Assistant working in a hospital setting. Your role is to:

1. Provide accurate medical information based on verified sources
2. Help interpret medical terminology in simple terms
3. Assist with understanding healthcare procedures and protocols
4. Direct users to appropriate medical resources
5. Maintain patient privacy and medical ethics

Important guidelines:
- Always clarify you're an AI assistant, not a doctor
- Recommend consulting healthcare professionals for specific medical advice
- Base responses on scientific evidence and reliable medical sources
- Keep responses clear, professional, and empathetic
- Never make definitive medical diagnoses

Knowledge base: You have access to WHO guidelines, CDC resources, Mayo Clinic data, and general healthcare information.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create tools
tools = [
    Tool(
        name="Healthcare Data Analysis",
        description="Tool for analyzing healthcare dataset using pandas operations",
        func=agent_test.run 
    ),
    Tool(
        name="Medical Literature Search",
        description="Tool for searching medical literature and guidelines",
        func=qa.invoke 
    )
]

# Initialize agent
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    prompt=prompt,
    llm=llm,
    verbose=True,
    max_iterations=10,
    early_stopping_method='generate',
    memory=conversational_memory,
    handle_parsing_errors=True #this is to give room to the llm to fix parsing errors, if any
)

class EnhancedHealthcareBot:
    def __init__(self):
        # Initialize environment variables
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")

        # System prompt from original code
        self.SYSTEM_PROMPT = """You are an advanced AI Healthcare Assistant working in a hospital setting. Your role is to:
        1. Provide accurate medical information based on verified sources
        2. Help interpret medical terminology in simple terms
        3. Assist with understanding healthcare procedures and protocols
        4. Direct users to appropriate medical resources
        5. Maintain patient privacy and medical ethics

        Important guidelines:
        - Always clarify you're an AI assistant, not a doctor
        - Recommend consulting healthcare professionals for specific medical advice
        - Base responses on scientific evidence and reliable medical sources
        - Keep responses clear, professional, and empathetic
        - Never make definitive medical diagnoses
        """

        # Initialize components
        self._components = {}
        self._initialize_base_components()

    def _initialize_base_components(self):
        """Initialize all required components"""
        # Load dataset
        self._components["df"] = pd.read_csv('dataset/healthcare_dataset.csv')

        # Initialize LLM
        self._components["llm"] = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            openai_api_key=self.openai_api_key
        )

        # Initialize memory
        self._components["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize vector store
        embed = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=self.openai_api_key
        )

        index = pc.Index("healthcare-qa-pdfs")
        self._components["vectorstore"] = Pinecone(index, embed.embed_query, "text")

        # Initialize QA chain
        self._components["qa_chain"] = RetrievalQA.from_chain_type(
            llm=self._components["llm"],
            chain_type="stuff",
            retriever=self._components["vectorstore"].as_retriever()
        )

        # Initialize tools
        self._components["tools"] = [
            Tool(
                name="Healthcare Data Analysis",
                description="Tool for analyzing healthcare dataset using pandas operations",
                func=agent_test.run
            ),
            Tool(
                name="Medical Literature Search",
                description="Tool for searching medical literature and guidelines",
                func=self._components["qa_chain"].run
            )
        ]

        # Initialize agent
        self._components["agent"] = initialize_agent(
            agent='chat-conversational-react-description',
            tools=self._components["tools"],
            prompt=self._get_prompt(),
            llm=self._components["llm"],
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=self._components["memory"]
        )

    def _get_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    def _query_healthcare_data(self, question: str) -> str:
        try:
            healthcare_tool = HealthcareDataFrameTool(df=self._components["df"])
            return healthcare_tool._run(question)
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def chat(self, user_input: str, use_voice: bool = False) -> dict:
        """Process user input and return response"""
        try:
            response = self._components["agent"].run(user_input)
            result = {
                'text': response,
                'voice': self.generate_voice(response) if use_voice else None,
                'error': None
            }
            return result
        except Exception as e:
            return {
                'text': f"Error: {str(e)}",
                'voice': None,
                'error': str(e)
            }

    def generate_voice(self, text: str, voice: str = "alloy") -> Optional[bytes]:
        """Generate voice response"""
        try:
            print(f"Generating voice for text: {text[:100]}...")  # Debug print
            
            if not text or len(text.strip()) == 0:
                print("Empty text provided")
                return None
                
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Ensure text is not too long (OpenAI has a limit)
            max_length = 4096
            if len(text) > max_length:
                text = text[:max_length]
                print(f"Text truncated to {max_length} characters")
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            content = response.content
            print(f"Voice generated successfully, content size: {len(content)} bytes")
            return content
            
        except Exception as e:
            print(f"Voice generation error: {str(e)}")
            print(f"API key valid: {bool(self.openai_api_key)}")  # Debug print
            return None

def chat_with_bot(use_voice=True):
    """Interactive chat interface with the healthcare bot"""
    bot = EnhancedHealthcareBot()
    audio_manager = AudioManager()
    
    print("Healthcare Bot: Hello! I'm your healthcare assistant. How can I help you today?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Healthcare Bot: Goodbye! Take care!")
            break
        
        response = bot.chat(user_input, use_voice=use_voice)
        print(f"Healthcare Bot: {response['text']}")
        
        if use_voice and response['voice']:
            audio_manager.play_audio(response['voice'])

if __name__ == "__main__":
    # Set use_voice=True to enable voice responses
    chat_with_bot(use_voice=True)