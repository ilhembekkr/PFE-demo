from pathlib import Path

import streamlit as st


from data_input import initialize_data
from core.ui import write_card, create_file_input


__version__ = "0.2"
_repository = "https://github.com/Coorsaa/automlbenchmark_shiny_app/"

def configure_streamlit():
    """Sets the streamlit page configuration."""
    st.set_page_config(
        page_title=f"AutoML-Benchmark Analysis App - v{__version__}",
        menu_items={
            "Get help": f"{_repository}",
            "Report a bug": f"{_repository}/issues/new",
        },
        layout="wide",
    )

st.write("# Review on end-to-end AutoML systems")

st.write("""

This app allows you to further inspect the results presented in the Master's thesis
"Review on end-to-end AutoML systems" by Ilhem Bekkar. The thesis discusses an empirical study about the different AutoML systems design and its limitations,
and contextualizes the results. While we provide some additional context in this app, we strongly encourage you to 
read the thesis before drawing any conclusions. 
""")



st.write("This app includes most of the figures of the thesis, with some"
         " additional controls which let you look at specific aspects of the data:")
# st.page_link("home.py", label=" * Figure 1: Benchmarking Suite")
st.page_link("pages/cd_diagram.py", label=" * Figure : Critical Difference Diagrams")
st.page_link("pages/performance.py", label=" * Figure: Scaled Performance Boxplots")
# st.page_link("home.py", label=" * Figure 5: Bradley-Terry Trees")
st.page_link("pages/inference.py", label=" * Figure: Model Inference Time")
st.page_link("pages/stability.py", label=" * Figure: Errors and training duration.")

st.write("We are still working on adding the other figures.")


write_card(
    body= "Click the three stacked dots in the top right, then navigate to"
          "`Settings` and select `Wide Mode`.",
    header="Plots too small?",
    icon="💡"
)



#create_file_input()

if __name__ == "__main__":
    initialize_data(Path("data/").expanduser())
