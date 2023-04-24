## Overview

The Agent creates an advertising tweet for you. Simply paste the website and the Agent will automatically generate engaging and concise posts that even includes hashtags!

This uses Langchain. The site is hosted by Streamlit.

---

## To run

Make sure to fill in the API keys for OpenAI and install all the dependencies. If not working, check the error log on streamlit site, they will let you know which ones you have to install.

```
python -m venv venv
source venv/bin/activate
pip install langchain
pip install openai
pip install faiss-cpu > /dev/null
pip install python-dotenv
pip install tiktoken
pip install streamlit
streamlit run agent.py

```

If streamlit is not working, checkout their [installation page](https://docs.streamlit.io/library/get-started/installation)

---

### Develop Log

23-04-22

1. Backend:
   - scrape data using webloader function
   - use FAISS to upload the data onto vectorspace
   - create tool (RetriveQA) so that the custom chain can use it to get data about the product

23-04-23

1. Frontend UI on streamlit
   - User input
2. Backend:
   - create custom prompt and custom LLM chain

---

### TODO

- [ ] Integrate Zapier
- [ ] Allow .txt and .csv files to upload

---
