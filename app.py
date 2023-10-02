import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import os
import warnings

cookies = EncryptedCookieManager(
    prefix="LUL/streamlit-cookies-manager/",
    password=os.environ.get("COOKIES_PASSWORD", "uDnda87,kGFdi&jh.kjsk/jk4DF369*^jhGks"),
)
warnings.filterwarnings("ignore")
import uuid
from datetime import datetime, timedelta

user_id = cookies.get('user_id')  # Attempt to retrieve the user ID cookie

if user_id is not None:
    st.write(f"Your user id is {user_id}")  # Display the user ID
else:
    user_id = str(uuid.uuid4())  # Generate a random user ID
    cookies['user_id'] = user_id  # Set the cookie
    st.write(f"Welcome! Your user id is {user_id}")  # Display the newly generated user ID
