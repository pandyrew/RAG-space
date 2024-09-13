import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import FAISSRetriever
from model import RAGModel

# set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# initialize retriever and model
logger.info("initializing retriever and model...")
retriever = FAISSRetriever()
model = RAGModel("llama2", retriever)
logger.info("retriever and model initialized successfully.")

# load documents
logger.info("loading documents...")
with open("data/fakefacts.txt", "r") as f:
    documents = f.readlines()
retriever.add_documents(documents)
logger.info(f"loaded {len(documents)} documents.")

class Query(BaseModel):
    text: str

@app.post("/generate")
async def generate(query: Query):
    # generate a response for the given query
    logger.info(f"received query: {query.text}")
    try:
        logger.debug("retrieving relevant documents...")
        retrieved_docs = retriever.retrieve(query.text)
        logger.debug(f"retrieved documents: {retrieved_docs}")
        
        logger.debug("generating response...")
        response = model.generate(query.text, retrieved_docs)
        logger.info("response generated successfully.")
        return {"response": response, "retrieved_docs": retrieved_docs}
    except Exception as e:
        logger.error(f"error generating response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"internal server error: {str(e)}")

if __name__ == "__main__":
    # start the fastapi server
    import uvicorn
    logger.info("starting fastapi server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)