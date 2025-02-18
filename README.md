![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Hospital Chatbot Project - Ironhack Final Project

## 🔖 Overview
This project is a **hospital chatbot** designed to help users:
- Get answers to hospital-related questions using a comprehensive healthcare dataset
- Analyze medical data and statistics from WHO guidelines
- Interact using both text and voice responses
- Access medical best practices and healthcare procedures

The chatbot leverages advanced AI technologies including **LangChain**, **OpenAI GPT-4**, and **Pinecone** for accurate, context-aware responses.

## ⚙️ Features
- **Smart Data Analysis**: Query and analyze healthcare dataset including patient records, billing information, and medical conditions
- **Voice Interaction**: Automatic voice responses using OpenAI's Text-to-Speech
- **Multimodal Interface**: Both command-line and Streamlit web interface
- **RAG Integration**: Access to WHO guidelines and healthcare statistics through PDF knowledge base
- **Contextual Memory**: Maintains conversation history for more coherent interactions

## 🛠️ Technical Stack
- **LangChain**: For agent creation and chain management
- **OpenAI**: GPT-4 for text generation, TTS for voice responses
- **Pinecone**: Vector database for document retrieval
- **Streamlit**: Web interface
- **Pygame**: Audio playback management
- **PyPDF2**: PDF document processing

## 📁 Project Structure
```plaintext
FinalProject/
├── .env                         # Environment variables configuration
├── dataset/
│   └── healthcare_dataset.csv   # Main healthcare dataset
├── healthcare_pdfs/            # Knowledge base documents
│   ├── 9789241513906-eng.pdf  # WHO quality health services guide
│   └── whohealthStat.pdf      # World health statistics 2024
├── Notebooks/                 # Jupiter Notebooks
│   └── notebook.ipynb        # Notebook
├── utils/
│   ├── __init__.py
│   └── audio_manager.py       # Audio playback system
├── src/
│   ├── main.py      #Bot Implementation          
│   └── app.py       #Streamlit Web Interface  
├── tests/
│   ├── app_test.py    #Streamlit Web Interface test
│   └── testscript.py  #Bot Implementation test
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up environment variables:
```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

### Running the Application
- For web interface:
```bash
streamlit run app.py
```
- For command line interface:
```bash
python main.py
```

## 🤖 Example Questions
- "What is the average billing amount for patients?"
- "How many patients are in the system?"
- "What are the most common medical conditions?"
- "Tell me about WHO's guidelines for quality healthcare"
- "What are the best practices for hospital management?"

## 🚀 Future Improvements

### Voice Interaction
- Add speech-to-text capability for voice input
- Implement voice activity detection for better interaction
- Add support for multiple languages in voice responses

### Data Analysis
- Implement advanced analytics for patient trends
- Add visualization tools for medical data

### UI/UX Enhancements
- Add dark mode support
- Implement mobile-responsive design
- Add data visualization dashboard

### AI Capabilities
- Integrate medical image analysis
- Add symptom checker functionality
- Implement medication interaction warnings
- Enhance context awareness across conversations
- Add support for medical document OCR

## 📨 Contact
For questions or suggestions:
- Email: solved.pt@gmail.com

