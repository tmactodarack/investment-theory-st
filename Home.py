import streamlit as st

from st_pages import show_pages_from_config, add_page_title
add_page_title()
show_pages_from_config()

from st_pages import Page, Section, show_pages, add_page_title
# add_page_title()
# show_pages(
#     [
#         Page("Home.py", "Home", "🏠"),
        
#         Section("sss", icon="📌"),
#         # Pages after a section will be indented
#         Page("pages/Candle_Chart.py",'ss', icon="💪"),
#         # # Unless you explicitly say in_section=False
#         # Page("Not in a section", in_section=False)
#     ]
# )

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.markdown('Run Era provides financial services such as macroeconomic forecast, stock valuation and portfolio optimization.')

st.markdown('*Select tool from the sidebar to start.*')