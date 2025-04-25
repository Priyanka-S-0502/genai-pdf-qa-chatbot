## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

### DESIGN STEPS:

#### STEP 1:
Before starting the implementation, ensure that all necessary libraries and dependencies are installed. This includes LangChain for processing the text, PyPDF2 (or similar) for reading PDF files, and an LLM like OpenAI for question-answering functionality.Install Necessary Libraries
#### STEP 2:
Use libraries like PyPDF2 to extract the text from the provided PDF document. The PDF extraction process should handle multiple pages and ensure that the text is clean and usable for further processing.
#### STEP 3:
Once the PDF text is extracted, it needs to be processed using LangChain’s tools, such as the TextSplitter and QuestionAnsweringChain, to handle large documents and provide accurate answers based on the content.

### PROGRAM:
```
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("tech.pdf")
pages = loader.load()

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-4', temperature=0)

# Build prompt
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
question = "Is probability a class topic?"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

result = qa_chain({"query": question})
print("Question: ", question)
print("Answer: ", result["result"])



```
### OUTPUT:

![Screenshot 2025-04-19 112031](https://github.com/user-attachments/assets/ed832117-ae9d-4a43-930b-076b29e26823)



### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
