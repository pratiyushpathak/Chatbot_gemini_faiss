import streamlit as st               
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import vectordb, chatbot01
import os
import pandas as pd

# def save_uploaded_file(uploaded_files, target_folder):
#     """
#     Save an uploaded file to a specified target folder.
 
#     Parameters:
#     - uploaded_file (BytesIO): The uploaded file object.
#     - target_folder (str): The path to the target folder where the file will be saved.
#     """
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
 
#     saved_file_paths = []
#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(target_folder, uploaded_file.name)
#         with open(file_path, "wb") as file:
#             file.write(uploaded_file.getbuffer())
#         saved_file_paths.append(file_path)
#     return saved_file_paths

def get_file_name(files):
    names =[]
    for file in files:
        names.append(file.split(".")[0])
        
    return names

def get_file_type(files):
    fType = []
    for file in files:
        # Extract file extension using the 'name' attribute
        filename = file.name
        fType.append(filename.split(".")[-1])
    return fType

    
def main():
    st.set_page_config("Chat PDF")
    st.header("Multi-PDF Chat using Gemini")

    if 'asked_questions' not in st.session_state:
        st.session_state.asked_questions = []
    st.subheader("Question")    
    user_question = st.text_input("Enter the question")
    
    st.sidebar.header("Select file type")
    folder_path = "./faiss_index/"
    files = os.listdir(folder_path)
    ftypes = get_file_name(files)
    selected_type = st.sidebar.selectbox("Select the File Type", set(ftypes))
    
    if selected_type is not None:
        st.sidebar.header("Select a file")
        folder_path = f"./faiss_index/{selected_type}"
        files = os.listdir(folder_path)
        file_names = get_file_name(files)
        selected_file = st.sidebar.selectbox("Select a file", set(file_names))
    
    if user_question:
        st.session_state.asked_questions.append(user_question)
        response = chatbot01.user_input(user_question,selected_type, selected_file)
        st.subheader("Answer")
        st.write(response["output_text"])
        # st.subheader("Doc Data")
        # doc_data = vectordb.get_query_data(user_question, selected_file, selected_type)
        # st.write(doc_data)
        
    

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        
        # Saving the uploaded file:
        
        # if pdf_docs is not None:
        #     save_button = st.button("Save Files")
        #     if save_button:
        #         target_folder = "uploaded_files"
        #         saved_file_paths = save_uploaded_file(pdf_docs, target_folder)
        #         st.success(f"Files saved successfully at: {', '.join(saved_file_paths)}")
            
        # ==========

        if st.button("Submit & Process") and pdf_docs:
            extension = get_file_type(pdf_docs)
            if extension[0] == "pdf":
                with st.spinner("Processing..."):
                    raw_text = vectordb.get_pdf_text(pdf_docs)
                    file_type = get_file_type(pdf_docs)
                    text_chunks = vectordb.get_text_chunks(raw_text)
                    vectordb.get_vector_store(text_chunks,file_type, pdf_docs[0].name)
                    vectordb.append_to_store(text_chunks,file_type)
                    st.rerun()
                    st.success("Done")
            
            elif extension[0] == "csv":
                with st.spinner("Processing..."):
                # raw_text = vectordb.get_csv_text()
                    file_type = get_file_type(pdf_docs)
                    text_chunks = vectordb.get_text_chunks(pd.read_csv(pdf_docs[0]).to_string())
                    vectordb.get_vector_store(text_chunks, file_type, pdf_docs[0].name)
                    vectordb.append_to_store(text_chunks,file_type)
                    st.rerun()
                    st.success("Done")
            
            else:
                st.warning("Please upload pdf or csv file")

     
        else:
            st.warning("Please select a file")
         
                
        
        
        # if st.button("Add data to existing index"):
        #     with st.spinner("Processing..."):
        #         raw_text = vectordb.get_pdf_text(pdf_docs)
        #         file_type = get_file_type(pdf_docs)
        #         text_chunks = vectordb.get_text_chunks(raw_text)
        #         vectordb.append_to_store(text_chunks, selected_file)
        #         st.success("Done")
        
        # st.subheader("Questions Asked:")
        # for idx, question in enumerate(st.session_state.asked_questions):
        #     st.write(f"{idx + 1}. {question}")
                
        
    
if __name__ == "__main__":
    main()




                

