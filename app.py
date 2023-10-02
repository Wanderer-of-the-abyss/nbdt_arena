
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import os
import warnings

cookies = EncryptedCookieManager(
    # This prefix will get added to all your cookie names.
    # This way you can run your app on Streamlit Cloud without cookie name clashes with other apps.
    prefix="LUL/streamlit-cookies-manager/",
    # You should really setup a long COOKIES_PASSWORD secret if you're running on Streamlit Cloud.
    password=os.environ.get("COOKIES_PASSWORD", "uDnda87,kGFdi&jh.kjsk/jk4DF369*^jhGks"),
)
warnings.filterwarnings("ignore")
import uuid
from datetime import datetime, timedelta

@st.cache_data
def gener():
   user_id = str(uuid.uuid4()) # generate a random user id

   cookies['user_id'] = user_id# set the cookie

user_id = cookies['user_id'] # get the cookie value
if user_id is not None:
    st.write(f"Your user id is {user_id}") # display the user id
else:
    gener()
