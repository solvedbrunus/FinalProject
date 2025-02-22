import streamlit as st
import requests
import os
import sys
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main import EnhancedHealthcareBot
from utils.audio_manager import AudioManager

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = EnhancedHealthcareBot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False

def play_audio(audio_bytes):
    """Play audio using AudioManager"""
    if audio_bytes:
        audio_manager = AudioManager()
        audio_manager.play_audio(audio_bytes)
        st.audio(audio_bytes, format='audio/mp3')

def main():
    st.set_page_config(
        page_title="Healthcare Assistant",
        page_icon="🏥",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("Settings")
        st.session_state.voice_enabled = st.checkbox("Enable Voice Response", value=True)
        
        st.header("About")
        st.markdown(
            """
            This chatbot interfaces with a
            [LangChain](https://python.langchain.com/docs/get_started/introduction)
            agent designed to answer questions about hospitals, patients,
            visits, physicians, and insurance payers in a simulated hospital system.
            
            The agent provides insights on healthcare best practices and metrics
            using retrieval-augmented generation (RAG).
            """
        )

        st.header("Example Questions")
        st.markdown("- What is the average duration in days by admission type?")
        st.markdown("- What was the total billing amount charged to each payer for 2023?")
        st.markdown("- How much was billed for patient with medicare?")
        st.markdown("- How are the most common admission types?")
        st.markdown("- What are the best practices to prevent diabetes?")
        st.markdown("- What are the most common procedures performed in the hospital?")
        st.markdown("- How can the quality of interventions improve healthcare pratice??")

    # Main chat interface
    st.title("Hospital System Chatbot")
    st.info("Ask me questions about patients, visits, insurance payers, hospitals.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "output" in message:
                st.markdown(message["output"])
            if "explanation" in message:
                with st.status("How was this generated", state="complete"):
                    st.info(message["explanation"])

    # Chat input
    if prompt := st.chat_input("What do you want to know?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "output": prompt})

        # Get bot response
        with st.spinner("Searching for an answer..."):
            response = st.session_state.chatbot.chat(
                prompt, 
                use_voice=st.session_state.voice_enabled
            )
            
            if response['error']:
                st.error(f"Error: {response['error']}")
            else:
                output_text = response['text']
                explanation = "Generated using advanced language model and healthcare knowledge base"
                
                st.chat_message("assistant").markdown(output_text)
                
                # Play voice if enabled
                if st.session_state.voice_enabled and response.get('voice'):
                    with st.spinner("Generating voice response..."):
                        play_audio(response['voice'])
                elif st.session_state.voice_enabled:
                    st.warning("Voice generation failed")

                with st.status("How was this generated", state="complete"):
                    st.info(explanation)

                st.session_state.messages.append({
                    "role": "assistant",
                    "output": output_text,
                    "explanation": explanation,
                    "voice": response.get('voice')
                })

if __name__ == "__main__":
    main()