# ShopEase-Assistant-
AI-Powered Customer Support Chatbot

# create python environment 
python -m venv chatbot_env

#Active the virtual environment
.\chatbot_env\Scripts\activate

# require Libaries
pip install pandas numpy sentence-transformers faiss-cpu datasets

# run the knowledge base creation script
python create_knowledge_base.py

# run main app using streamlit
streamlit run app.py
