# Creating a Korean Chatbot using RAG with Amazon Bedrock

Chatbot that performs questions/answering using the Anthropic Claude LLM (Large Language Models) model of [Amazon Bedrock](https://aws.amazon.com/ko/bedrock/) [Knowledge Database] Implemented using (https://aws.amazon.com/ko/about-aws/whats-new/2023/09/knowledge-base-amazon-bedrock-models-data-sources/). A large-scale language model (LLM) pretrained with large amounts of data can find the closest answer according to the context and answer even untrained questions. Additionally, if you use [RAG(Retrieval-Augmented Generation)](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html), LLM can The impact of hallucination can be reduced, and [fine tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-fine-tuning.html) You can utilize the latest data as provided. RAG is one of the [prompt engineering](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-prompt-engineering.html) technologies that uses vector store as a knowledge database. I'm doing it.

Vector store can store and retrieve unstructured content such as images, text documents, and audio. In particular, in the case of large-scale language models (LLM), the semantic meaning of texts can be expressed as a vector using embedding, so the closest answer to the question can be found through semantic search. Here, [Faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started), a representative in-memory vector store, and [Amazon OpenSearch](https://medium. com/@pandey.vikesh/rag-ing-success-guide-to-choose-the-right-components-for-your-rag-solution-on-aws-223b9d4c7280) and Kendra, a fully managed search service. Implements RAG.

## Architecture Overview

The overall architecture is as follows. Users read the resources needed for web pages from [Amazon S3](https://aws.amazon.com/ko/s3/) through Amazon CloudFront. When a user logs in to the Chatbot web page, the chat history stored in Amazon DynamoDB is loaded using the user ID. Afterwards, when the user enters a message, a query is queried to LLM using WebSocket, and an appropriate response is obtained using chat history read from DynamoDB and related documents (Relevant docs) read from Vector Database that provides RAG. To implement this RAG, an application was developed by [using LangChain](https://python.langchain.com/docs/get_started/introduction.html), and the infrastructure that provides Chatbot is [AWS CDK](https:// Distributed through aws.amazon.com/ko/cdk/).

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/45ef56fb-110e-4c42-bc39-846977402438)


A detailed step-by-step explanation is as follows.

Step 1: When a user accesses the CloudFront address using a browser, Amazon S3 delivers files such as HTML, CSS, and JS. At this time, log in and enter the chat screen.

Step 2: The client requests chat history using the ‘/history’ API using the user ID. This request goes through API Gateway and is delivered to lambda-history. Afterwards, the chat history is retrieved from DynamoDB and then delivered to the user through API Gateway and lambda-history.

Step 3: When the client attempts a WebSocket connection to API Gateway, a WebSocket connection event is delivered to lambda-chat-ws through API Gateway. Afterwards, when the user enters a message, the message is delivered to lambda-chat-ws through API Gateway.

Step 4: lambda-chat-ws reads the existing chat history from DynamoDB using the user ID and stores it in the chat memory.

Step 5: lambda-chat-ws searches relevant docs for RAG.

Step 6: lambda-chat-ws forwards the user’s questions, chat history, and relevant docs to Amazon Bedrock’s Enpoint.

Step 7: When the user's questions and chat history are sent to Amazon Bedrock, an appropriate answer is delivered to the user using Anthropic's Claude LLM. In this case, you can use stream to show the answer to the user before the answer is completed.

The sequence diagram at this time is as follows.

![seq-chat](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/13818855-0a63-4d5e-9f9b-1b98245f80b6)


## Main configuration

### Connecting Bedrock to LangChain

Currently (September 2023) you can use AWS Bedrock commercially without restrictions. You can develop applications with LangChain by importing [Bedrock](https://python.langchain.com/docs/integrations/providers/bedrock). Here we use us-east-1 as bedrock_region.

Define bedrock client as follows. The service name is “bedrock-runtime”.

```python
import boto3
from langchain.llms.bedrock import Bedrock

boto3_bedrock = boto3.client(
     service_name='bedrock-runtime',
     region_name=bedrock_region;
     config=Config(
         retries = {
             'max_attempts': 30
         }
     )
)

llm = Bedrock(
     model_id=modelId,
     client=boto3_bedrock,
     streaming=True,
     callbacks=[StreamingStdOutCallbackHandler()],
     model_kwargs=parameters)
```

Here the parameters are as follows.

```python
def get_parameter(modelId):
     if modelId == 'amazon.titan-tg1-large' or modelId == 'amazon.titan-tg1-xlarge':
         return {
             "maxTokenCount":1024,
             "stopSequences":[],
             "temperature":0,
             "topP":0.9
         }
     elif modelId == 'anthropic.claude-v1' or modelId == 'anthropic.claude-v2':
         return {
             "max_tokens_to_sample":8191, #8k
             "temperature":0.1,
             "top_k":250,
             "top_p": 0.9;
             "stop_sequences": [HUMAN_PROMPT]
         }
parameters = get_parameter(modelId)
```

Bedrock's supported models can be searched using list_foundation_models() of [service name is "bedrock"](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock.html).

```python
bedrock_client = boto3.client(
     service_name='bedrock',
     region_name=bedrock_region,
)
modelInfo = bedrock_client.list_foundation_models()
print('models: ', modelInfo)
```



###Embedding

Embedding is done using [BedrockEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/bedrock). 'amazon.titan-embed-text-v1' stands for Titan Embeddings Generation 1 (G1) and supports 8k tokens.

```python
bedrock_embeddings = BedrockEmbeddings(
     client=boto3_bedrock,
     region_name = bedrock_region;
     model_id = 'amazon.titan-embed-text-v1'
)
```

## Knowledge Database definition

Here we learn about OpenSearch, Faiss, and Kendra as a Knowledge Database.

### OpenSearch

Define a vector store using [OpenSearchVectorSearch](https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.opensearch_vector_search.OpenSearchVectorSearch.html). The default engine here is nmslib, but you can select faiss or lucene depending on your needs.

```python
from langchain.vectorstores import OpenSearchVectorSearch

vectorstore = OpenSearchVectorSearch(
     index_name = 'rag-index-'+userId+'-*',
     is_aoss = False;
     ef_search = 1024, # 512 (default)
     m=48;
     #engine="faiss", # default: nmslib
     embedding_function = bedrock_embeddings,
     opensearch_url=opensearch_url,
     http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
)
```

Data can be added to the vector store using OpenSearch using add_documents() as shown below. Here, in order to apply a personalized RAG using an index, the index is defined as userId and requestId as shown below, and then a new vector store is defined and used.

```python
new_vectorstore = OpenSearchVectorSearch(
     index_name="rag-index-"+userId+'-'+requestId,
     is_aoss = False;
     #engine="faiss", # default: nmslib
     embedding_function = bedrock_embeddings,
     opensearch_url = opensearch_url,
     http_auth=(opensearch_account, opensearch_passwd),
)
new_vectorstore.add_documents(docs)
```

Related documents (relevant docs) can be searched as follows.

```python
relevant_documents = vectorstore.similarity_search(query)
```

### Faiss

We define Faiss as a vector store as shown below. Here, Faiss is an in-memory vectore store that can only be used while the instance is maintained. We also use add_documents() to insert data into the faiss vector store. Since you can search while data is entered, check isReady as shown below.

```python
vectorstore = FAISS.from_documents( # create vectorstore from a document
     docs, #documents
     bedrock_embeddings #embeddings
)
isReady = True

vectorstore.add_documents(docs)
```

Related documents (relevant docs) can be searched as follows.

```python
query_embedding = vectorstore.embedding_function(query)
relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
```

### Kendra

Kendra does not require embedding, so specify the retriever by setting the index_id as shown below.

```python
from langchain.retrievers import AmazonKendraRetriever
kendraRetriever = AmazonKendraRetriever(index_id=kendraIndex)
```

Use [kendraRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.kendra.AmazonKendraRetriever.html?highlight=kendraretriever#langchain.retrievers.kendra.AmazonKendraRetriever) to find related documents as below: You can search.

```python
relevant_documents = kendraRetriever.get_relevant_documents(query)
```

### RAG implementation, including related documentation

The actual result is [RetrievalQA](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html?highlight=retrievalqa#langchain.chains.retrieval_qa.base.RetrievalQA ) to obtain it.

relevant_documents = vectorstore.similarity_search(query)

```python
qa = RetrievalQA.from_chain_type(
     llm=llm,
     chain_type="stuff",
     retriever=retriever,
     return_source_documents=True;
     chain_type_kwargs={"prompt": PROMPT}
)
result = qa({"query": query})
```

Here, retriever is defined as follows. Here, kendra's retriever is [AmazonKendraRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.kendra.AmazonKendraRetriever.html?highlight=kendraretriever#langchain.retrievers.kendra.AmazonKendraRetriever). Define, opensearch and faiss [VectorStore](https://api.python.langchain.com/en/latest/schema/langchain.schema.vectorstore.VectorStore.html?highlight=as_retriever#langchain.schema.vectorstore.VectorStore .as_retriever).

```python
if rag_type=='kendra':
     retriever = kendraRetriever
elif rag_type=='opensearch' or rag_type=='faiss':
     retriever = vectorstore.as_retriever(
         search_type="similarity",
         search_kwargs={
             "k": 3
         }
     )
```


### Display Reference

As shown below, kendra extracts reference information from the doc metadata. Here, the file name is obtained using doc.metadata['title'], and the page is obtained using doc.metadata['document_attributes']['_excerpt_page_number']. The URL is constructed by combining the cloudfront URL and the S3 bucket key and object. opensearch and faiss obtain the file name, page number, and path (URL path) through 'name', 'page', and 'url' of metadata.

```python
def get_reference(docs, rag_type):
     if rag_type == 'kendra':
         reference = "\n\nFrom\n"
         for doc in docs:
             name = doc.metadata['title']
             url = path+name

             if doc.metadata['document_attributes']:
                 page = doc.metadata['document_attributes']['_excerpt_page_number']
                 reference = reference + f"{page}page in <a href={url} target=_blank>{name}</a>\n"
             else:
                 reference = reference + f"in <a href={url} target=_blank>{name}</a>\n"
     else:
         reference = "\n\nFrom\n"
         for doc in docs:
             name = doc.metadata['name']
             page = doc.metadata['page']
             url = doc.metadata['url']
        
             reference = reference + f"{page}page in <a href={url} target=_blank>{name}</a>\n"
        
     return reference
```


### Creation of Prompt

To get an answer to a question using both RAG and chat history, [ConversationalRetrievalChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html ?highlight=conversationalretrievalchain#langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain), or use retrivalQA after creating a new prompt with the history and current prompt.

Here's how to create the current prompt considering chat history. Here we create a new question by including "rephrase the follow up question" in the prompt template.

```python
generated_prompt = get_generated_prompt(text)

def get_generated_prompt(query):
     condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

     Chat History:
     {chat_history}
     Follow Up Input: {question}
     Standalone question:"""
     CONDENSE_QUESTION_PROMPT = PromptTemplate(
         template = condense_template, input_variables = ["chat_history", "question"]
     )
    
     chat_history = extract_chat_history_from_memory()
    
     question_generator_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
     return question_generator_chain.run({"question": query, "chat_history": chat_history})
```

Afterwards, you can obtain the generated questions and the results of applying RAG using RetrievalQA.

## Save conversation in memory

### When not using RAG

lambda-chat-ws uses the userId of the incoming message to check whether there is a conversation history (memory_chat) previously stored in map_chat. If there is no chat history, as shown below [ConversationBufferMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html?highlight=conversationbuffermemory#langchain.memory.buffer.ConversationBufferMemory) Set memory_chat to . Here, Anthropic Claude sets "Human" and "Assistant" as the names of human and ai. When requesting a response from LLM, ConversationChain is used.

```python
map_chat = dict()

if userId in map_chat:
     memory_chat = map_chat[userId]
else:
     memory_chat = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant')
     map_chat[userId] = memory_chat
conversation = ConversationChain(llm=llm, verbose=False, memory=memory_chat)
```

You can simply limit the number of conversations to k using [ConversationBufferWindowMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html).


Here, when using Faiss, if there is no conversation history, RAG cannot be used, so it is applied as above.

### When to use RAG

When using RAG, use [ConversationBufferMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html?highlight=conversationbuffermemory#langchain.memory.buffer.ConversationBufferMemory). Specify chat memory as shown below. After the conversation ends, update the new chat dialog using add_user_message() and add_ai_message().

```python
map_chain = dict()

if userId in map_chain:
     memory_chain = map_chain[userId]
else:
     memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
     map_chain[userId] = memory_chain

msg = get_answer_from_conversation(text, conversation, convType, connectionId, requestId)

memory_chain.chat_memory.add_user_message(text) # append new diaglog
memory_chain.chat_memory.add_ai_message(msg)

def get_answer_from_conversation(text, conversation, convType, connectionId, requestId):
     conversation.prompt = get_prompt_template(text, convType)
     stream = conversation.predict(input=text)
     msg = readStreamMsg(connectionId, requestId, stream)

     return msg
```

Here, the stream can deliver a message to a client using WebSocket in the following manner.

```python
def readStreamMsg(connectionId, requestId, stream):
     msg = ""
     if stream:
         for event in stream:
             msg = msg + event

             result = {
                 'request_id': requestId,
                 'msg': msg
             }
             sendMessage(connectionId, result)
     print('msg: ', msg)
     return msg
```

Here, sendMessage(), which sends a message to the client, is as follows. Here, the message is sent to API Gateway, the endpoint of WebSocket, using boto3's [post_to_connection](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigatewaymanagementapi/client/post_to_connection.html). do.

```python
def sendMessage(id, body):
     try:
         client.post_to_connection(
             ConnectionId=id;
             Data=json.dumps(body)
         )
     except:
         raise Exception ("Not able to send a message")
```

### Implementing Infrastructure with AWS CDK

[CDK implementation code](./cdk-qa-with-rag/README.md) provides detailed information on how to define infrastructure with Typescript.

## Try it yourself

### Preparation

In order to use this solution, the following preparations must be made in advance.

- [AWS AcCreate count](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)


### Infrastructure installation using CDK
Proceed with installing infrastructure with CDK according to [Infrastructure Installation](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md).


### Execution result

After downloading [fsi_faq_ko.csv](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/fsi_faq_ko.csv), select the file icon to upload and it will be saved to the Knowledge Database. . Afterwards, the file contents are summarized and displayed so that you can check them as shown below.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/a0b3b5b8-6e1e-4240-9ee4-e539680fa28d)

In the chat window, “Can I use the easy inquiry service in English?” Enter. The result at this time is “No.” The result at this time is as follows.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c7aeca05-0209-49c3-9df9-7e04026900f2)

The chat window says, “Transfer is not possible. What should I do?” Enter and check the results.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/56ad9192-6b7c-49c7-9289-b6a3685cb7d4)

In the chat window, “What is the joint certificate window issuance service?” If you type, you can get the result as below.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/95a78e6a-5a78-4879-98a1-a30aa6f7e3d5)

#### Translation

Enter "Thank you for using Amazon Bedrock. You can enjoy a comfortable conversation, and you can summarize by uploading a file." and check the translation result.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/818662e1-983f-44c2-bfcf-e2605ba7a1e6)

#### Extracted Topic and sentiment

“The meal is good value for money. The location is good and the sky lounge barbecue/night view is the best. Disappointing points: The underground parking lot is cramped. Traffic in front of the hotel is so complicated that it is difficult to use nearby facilities. / I need a way to get to the Han River / I need a way to get to nearby facilities, etc.” and check the results.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/8c38a58b-08df-4e9e-a162-1cd8f542fb46)

#### Information extraction

“John Park. Solutions Architect | Enter WWCS Amazon Web Services Email: john@amazon.com Mobile: +82-10-1234-5555 and check whether the email is extracted.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/f613e86e-b08d-45e8-ac0e-71334427f450)

#### Deleting personally identifiable information (PII)

An example of deletion of PII (Personal Identification Information) is below. Enter "John Park, Ph.D. Solutions Architect | WWCS Amazon Web Services Email: john@amazon.com Mobile: +82-10-1234-4567" to get the text with name, phone number, and address removed. . For the prompt, see [PII](https://docs.anthropic.com/claude/docs/constructing-a-prompt).

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/a77d034c-32fc-4c84-8054-f4e1230292d6)

#### Fix sentence errors

Enter the sentence with the error as "To have a smooth conversation with a chatbot, it is better for usabilities to show responses in a stream-like, conversational maner rather than waiting until the complete answer."

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/55774e11-58e3-4eb4-b91c-5b09572456bd)

Enter "For smooth interaction with Chatbot, it is better to show the user's question in a stream rather than wait until the user's question is fully answered." and check the result.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/7b098a29-9bf5-43bf-a32f-82c94ccd04eb)

#### Complex question (step-by-step)

Type "I have two pet cats. One of them is missing a leg. The other one has a normal number of legs for a cat to have. In total, how many legs do my cats have?" and see the results.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c1bf6749-1ce8-44ba-81f1-1fb52e04a2e8)


Enter your question as "I have two cats. One of them is missing a leg. The other has the number of legs a cat should normally have. In total, how many legs do my cats have?" and check the results.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/992c8385-f897-4411-b6cf-b185465e8690)

#### Talk to children (Few shot example)

You must answer questions according to the person you are speaking with. For example, when [General Conversation] asks the question, “Will Santa bring presents on Christmas?”, the answer is as follows.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/4624727b-addc-4f5d-8f3d-94f358572326)

[9. Switch to [Child Conversation (few shots)]. I ask the same question. I was able to give an appropriate answer tailored to the other person.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/cbbece5c-5476-4f3b-89f7-c7fcf90ca796)


## Organizing resources

If you are no longer using the infrastructure, you can delete all resources as shown below.

1) Connect to [API Gateway Console](https://ap-northeast-2.console.aws.amazon.com/apigateway/main/apis?region=ap-northeast-2) and click “rest-api-for- Delete "stream-chatbot", "ws-api-for-stream-chatbot".

2) Access [Cloud9 console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/) and delete everything using the command below. .


```text
cdk destroy --all
```

## conclusion

We implemented a chatbot that performs questions and answers using Amazon Bedrock and vector store in the AWS Seoul region. Amazon Bedrock allows you to choose one of several large-capacity language models to use. Here, we implemented RAG operation using Amazon Titan and were able to solve the hallucination problem of a large language model. Additionally, LangChain was used to develop the chatbot application, and AWS CDK was used as IaC (Infrastructure as Code). It is expected that large-capacity language models will be effectively utilized in various applications in the future.

You will be treated well. By developing large-scale language models using Amazon Bedrock, you can easily integrate with existing AWS infrastructure and effectively develop various applications.



## Reference

[Claude - Constructing a prompt](https://docs.anthropic.com/claude/docs/constructing-a-prompt)