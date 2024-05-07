from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import vectordb

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,if the question is about you or a greeting then greet the user as well or if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. If the question is numeric then give the correct solution\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    # if question.lower() in predefined_responses:
    #         return predefined_responses[question.lower()]

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm=vectordb.llm, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question,file_type, index):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = faiss.FAISS.load_local(f"faiss_index/{file_type}", embeddings,allow_dangerous_deserialization=True, index_name=index)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    return response
