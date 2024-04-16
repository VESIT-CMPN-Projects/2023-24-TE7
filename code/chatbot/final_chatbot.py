import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import os
import requests
from urllib.parse import urlparse
import pickle
import pandas as pd
from PIL import Image

vector_store = []

os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'


st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon=":blue_book:",
)


# Function to get text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    print(vector_store)
    return vector_store


# Function to create conversational chain from vector store
prompt_template = """
Make sure to provide all the details, also make sure that the formatting of the answer is nice, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer, if the answer contains numerical data, then also give the units like $ or million or billion based on what is given in the context, if the user requests the answer in tabular format, please provide the answer accordingly in a perfectly formatted table, If the question is of logical reasoning and open-ended, please give logical answers, also justify the alignment of the answers Give all the answer in proper formatting, if the user asks to answer in bullet points, then answer in bullet points ensuring each point starts from a new line.\n\n
Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""


# Function to create conversational chain from vector store
def get_conversational_chain(vector_store, prompt_template):
    llm = GooglePalm()
    # Load conversational chain with prompt
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    )
    return conversation_chain


# Function to handle user input
def user_input(user_q):
    print("here")
    response = st.session_state.conversation({'question': user_q})
    st.session_state.chatHistory = response['chat_history']
    st.rerun()


# Function to process uploaded PDFs
def process_pdf(pdf_path):
    main_text = ""
    with pdf_path as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            main_text += page.extract_text()
        text_chunks = get_text_chunks(main_text)
        vector_store = get_vector_store(text_chunks)
        st.session_state.conversation = get_conversational_chain(vector_store, prompt_template)
        st.session_state.faq_conversation = get_conversational_chain(vector_store, prompt_template)
    return f"Processed PDF: {os.path.basename(pdf_path.name)}"


# Function to search for PDFs using Google Custom Search API
def search_for_pdfs(query, num_results=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    cse_id = 'YOUR_CSE_ID'
    api_key = 'YOUR_CUSTOM_CS_API_KEY'
    
    params = {
        'q': f"{query} annual public report filetype:pdf",
        'num': num_results,
        'cx': cse_id,
        'key': api_key,
    }
    
    response = requests.get(search_url, params=params)
    data = response.json()

    pdf_results = []
    if 'items' in data:
        for item in data['items']:
            pdf_url = item['link']
            pdf_name = os.path.basename(urlparse(pdf_url).path)
            pdf_results.append((pdf_name, pdf_url))

    return pdf_results


def save_processed_data(text_chunks):
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(text_chunks, f)


# Function to load processed text data
def load_processed_data():
    if os.path.exists('processed_data.pkl'):
        with open('processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return None
  
    
import plotly.graph_objects as go

def save_to_csv(df, filename='data.csv'):
    df.to_csv(filename, index=False)
   
    
def generate_graph_from_csv():
    # Read the CSV file
    df = pd.read_csv("data.csv")

    # Set the keywords as the index
    df.set_index(df.columns[0], inplace=True)

    # Convert string representations of monetary values to numeric values
    for col in df.columns:
        if df[col].dtype == object:  # Check if column dtype is object (string)
            # Replace dollar signs, commas, and percentage signs with empty string
            df[col] = df[col].str.replace('[\$,%,million,billion,M,B,m,b,Million,Billion]', '', regex=True)
            # Replace '---' with NaN (Not a Number)
            df[col] = df[col].replace('[---, ------, --, -, ----, -----]', np.nan)
            # Convert to float
            df[col] = df[col].astype(float)

    # Plotting the graph using Plotly
    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[column],
            name=column,
            hovertemplate='<b>%{x}</b><br>%{customdata}',
            customdata=df[column].apply(lambda x: f"${x:,.2f}" if not np.isnan(x) else "Not available"),
        ))

    # Update layout to keep y-axis empty and show hover information
    fig.update_layout(
        barmode='group',
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        xaxis_title='Fiscal Year',
        yaxis=dict(
            title='',
            showticklabels=False
        ),
        legend=dict(
            title='Metrics',
            font=dict(
                size=12
            )
        )
    )

    # Displaying the graph
    st.plotly_chart(fig)


# Function to load and resize images
def load_image(image_path, size=(50,50)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image


# Load images
human_image = load_image("human_icon.png", size=(100,100))
chatgpt_image = load_image("bot.png", size=(100,100))


# Main function
def main():
    
    st.header("Chat with Annual Public Reports 💲")


    # Check session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None


    # Sidebar for settings
    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload and Process PDFs")


    # Upload PDFs
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    faq_responses = {}
    # Process uploaded PDFs
    if pdf_docs and st.sidebar.button("Process PDFs"):
        with st.spinner("Processing PDFs"):
            for pdf in pdf_docs:
                st.write(process_pdf(pdf))
        with st.spinner("Generating FAQ's"):
            st.session_state.faq_displayed = True
            st.subheader("Here some FAQ's")
            faq_questions = {
                "Financial Performance": ["What were the total revenues and net profits for the year?", 
                                        "How did the company perform financially in the last fiscal year?"],
                "Operational Highlights": ["Can you provide key operational highlights mentioned in the report?"],
                "Risk Mitigation": ["How does the company plan to mitigate potential risks or challenges?"],
                "Strategic Goals": ["What are the company's plans and strategic goals for the upcoming fiscal year?"],
                "Sustainability Initiatives": ["Can you provide details about the company's sustainability initiatives?"]
            }


            # Display FAQ questions and corresponding chatbot answers
            for category, questions in faq_questions.items():
                st.subheader(category)
                for question in questions:
                    expander = st.expander(f"Q: {question}")
                    with expander:
                        response = st.session_state.faq_conversation({'question': question})
                        if response['chat_history']:
                            answer = response['chat_history'][-1].content
                            st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {answer}</p>',unsafe_allow_html=True)
                        else:
                            st.write("A: No answer found")
            st.session_state.faq_conversation = None


    # Search bar for PDFs
    company_name = st.sidebar.text_input("Enter Company Name", key="company_name")
    year = st.sidebar.text_input("Enter Year", key="year")
    search_button_clicked = st.sidebar.button("Search")


    # Search for PDFs based on company name and year
    if search_button_clicked:
        if company_name and year:
            search_results = search_for_pdfs(company_name + " " + year)
            if search_results:
                st.sidebar.subheader("Search Results:")
                for pdf_name, pdf_url in search_results:
                    st.sidebar.markdown(
                        f'<a href="{pdf_url}" target="_blank" style="display: block; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc; text-decoration: none; color: #333; background-color: #f9f9f9;">{pdf_name}</a>',
                        unsafe_allow_html=True
                    )
            else:
                st.sidebar.write("No results found.")
        else:
            st.sidebar.write("Please enter both company name and year.")


    # if 'question_submitted' not in st.session_state:
    #     st.session_state.question_submitted = False
# Handle user input


    if st.session_state.chatHistory:
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.image(human_image, width=40)
                with col2:
                    st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-align: left;">{message.content}</p>', unsafe_allow_html=True)
            else:
                if "|" in message.content and "---" in message.content:
                    # If the response contains table formatting, format it into a proper table
                    rows = [row.split("|") for row in message.content.split("\n") if row.strip()]
                    
                    # Remove empty first and last columns
                    if rows and len(rows[0]) > 2:
                        rows = [row[1:-1] for row in rows]

                    # Remove rows filled with '--- --- ---'
                    character = '-'

                    # Filter out rows where all cells contain only the specified character
                    rows = [row for row in rows if not all(cell.strip() == character * len(cell.strip()) for cell in row)]

                    if len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows):
                        # Check for empty column names and replace them with a default name
                        columns = [col.strip() if col.strip() else f"Column {i+1}" for i, col in enumerate(rows[0])]
                        df = pd.DataFrame(rows[1:], columns=columns)
                        st.write(df)
                    
                        
                        save_to_csv(df)
                        generate_graph_from_csv()
                        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)

                        
                    else:
                        col1, col2 = st.columns([1, 8])
                        with col1:
                            st.image(chatgpt_image)
                        with col2:
                            
                            st.write(message.content)
                        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)

                else:
                    col1, col2 = st.columns([1, 8])
                    with col1:
                        st.image(chatgpt_image)
                    with col2:
                        
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{message.content}</p>', unsafe_allow_html=True)
                    st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
        



    user_question = st.text_input("Ask a Question from the PDF Files", key="pdf_question")


    if st.button("Get Response") and user_question :
        user_q =  user_question
        user_input(user_q)


if __name__ == "__main__":
    main()