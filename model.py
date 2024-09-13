from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class RAGModel:
    def __init__(self, model_name, retriever):
        # initialize the language model, retriever, and prompt template
        self.llm = Ollama(model=model_name, base_url="http://localhost:11434")
        self.retriever = retriever
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Analyze the following context and answer the question concisely. Use bullet points for multiple facts. Do not add any conversational elements.

Context: {context}

Question: {question}

Answer:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate(self, query, retrieved_docs):
        # generate a response based on the query and retrieved documents
        context = "\n".join(retrieved_docs)
        response = self.chain.run(context=context, question=query)

        # post-process the response to remove any remaining conversational elements
        response = response.replace("Hi there!", "").replace("I hope this information helps!", "").strip()
        return response