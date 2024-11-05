# Chain-of-Thought-RAG
This is a RAG chatbot made with local LLMs. I used Ollama for local LLMs, and the user interface was made with the help of Streamlit.

To run the chatbot user have to install Ollama in their machine.
steps to install Ollama in the machine:
  1. Go to the [link](https://ollama.com/download) and download ollama.
  2. For Windows install the .exe file.
  3. For Linux, open the terminal and type <code>curl -fsSL https://ollama.com/install.sh | sh </code>.
  4. After installing, users need to install two models used in the RAG, which are LLama3.2 and nomic.
  5. To download the models open cmd for Windows or terminal for Linux:
         llama3.2 : ollama pull llama3.2
         nomic: ollama pull nomic-embed-text
After configuring ollama, the user can check if ollama is properly installed by typing [link](https://127.0.0.1:11434) in the browser.

The GPU requirement for the whole process is  4574 MiB.

After setting up the ollama server, user have to install python packages:
<code> pip install -r requirements.txt </code>
After installing the requirements run <code>python db.py</code> to create the vector database.
Then type <code>streamlit run app.py</code> to run the app.
