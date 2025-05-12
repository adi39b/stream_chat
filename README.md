Okay, here is the complete code for the Streamlit application, incorporating `st.write_stream` and minimizing comments, ready for export.

**Project Structure:**

```
your_app_directory/
├── SmartAssist.py
├── research_workflow.py
└── pages/
    ├── 02_Run_History.py
    ├── 03_Settings.py
    └── 04_Global_Records.py
```

**1. `research_workflow.py`**

```python
# research_workflow.py
import time
import random
import pandas as pd
import numpy as np

_DUMMY_DATA = {
    "Company Overview": lambda name: [
        f"Researching general overview for {name}...", "\n\n",
        f"{name} is a notable entity within its industry, founded around {random.randint(1950, 2010)}. ",
        "It boasts a significant presence across major global markets, including North America, Europe, and the APAC region. ",
        "The core philosophy revolves around continuous innovation and ensuring high levels of customer engagement and satisfaction. ",
        f"Recent strategic moves involve deeper penetration into developing economies and substantial investments in proprietary R&D projects."
    ],
    "Financial Highlights": lambda name: [
        f"Analyzing financial performance indicators for {name}...", "\n\n",
        f"The latest disclosed annual revenue stands at approximately ${random.uniform(1, 50):.2f} billion USD. ",
        f"Net profit margin showed a {random.choice(['positive trend', 'slight decrease', 'stable performance'])} compared to the previous cycle, changing by {random.uniform(1, 10):.1f}%. ",
        "Key financial metrics suggest a robust operational foundation. ",
        f"Stock Ticker (Illustrative): {name[:1].upper()}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}. ", "\n\n",
        pd.DataFrame(
            np.random.randn(3, 5) * 100,
            columns=["Metric A", "Metric B", "Metric C", "Metric D", "Metric E"],
            index=[f"Q{i} FY{time.strftime('%y')}" for i in range(1,4)]
        )
    ],
    "Recent News & Developments": lambda name: [
        f"Aggregating recent news and significant developments for {name}...", "\n\n",
        f"* **(Source: Global Business Wire, {time.strftime('%Y-%m-%d')})**: {name} unveiled a strategic alliance with 'Innovate Solutions Inc.' to explore next-gen technologies.\n",
        f"* **(Source: Finance Today, {time.strftime('%Y-%m-%d', time.localtime(time.time()-86400))})**: Quarterly results surpassed market forecasts, driven by strong performance in the services division.\n",
        f"* **(Source: Tech Chronicle, {time.strftime('%Y-%m-%d')})**: Speculation surrounds {name}'s potential launch of a groundbreaking product suite in the upcoming fiscal quarter.\n"
    ],
    "References & Citations": lambda name: [
        "Compiling sources and citations...", "\n\n",
        "[1] Global Business Wire Archives. (Accessed: Internal Database).\n",
        "[2] Finance Today Market Reports. (Accessed: Internal Database).\n",
        "[3] Tech Chronicle Industry Analysis. (Accessed: Internal Database).\n",
        f"[4] {name} Investor Relations Portal (Simulated Data). Fiscal Year {time.strftime('%Y')}. (Accessed: Internal Database)."
    ]
}

def stream_section_data(company_name, section_name):
    """Generator for yielding chunks of data for a specific research section."""
    if section_name in _DUMMY_DATA:
        content_generator = _DUMMY_DATA[section_name]
        content_parts = content_generator(company_name)
        is_first_string_yield = True
        for part in content_parts:
            if isinstance(part, str):
                words = part.split(' ')
                if is_first_string_yield:
                    time.sleep(random.uniform(0.1, 0.3))
                    is_first_string_yield = False
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
                    time.sleep(random.uniform(0.01, 0.05))
            else:
                time.sleep(0.5)
                yield part
                time.sleep(0.2)
                is_first_string_yield = True
    else:
        yield f"Content generation for section '{section_name}' is not implemented."
        time.sleep(0.1)

```

**2. `SmartAssist.py`**

```python
# SmartAssist.py
import streamlit as st
import time
from research_workflow import stream_section_data

st.set_page_config(
    page_title="SmartAssist Research",
    page_icon="💡",
    layout="wide"
)

st.title("💡 SmartAssist - Company Research")
st.caption(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize session state variables
if 'research_running' not in st.session_state:
    st.session_state.research_running = False
if 'company_for_report' not in st.session_state:
    st.session_state.company_for_report = ""
if 'report_sections' not in st.session_state:
    st.session_state.report_sections = {}
if 'company_input_value' not in st.session_state:
    st.session_state.company_input_value = ""

# --- UI Elements ---
def update_input_state():
    st.session_state.company_input_value = st.session_state.company_input_field

company_input = st.text_input(
    "Enter Company Name:",
    value=st.session_state.company_input_value,
    placeholder="e.g., Google, Acme Corporation",
    key="company_input_field",
    on_change=update_input_state
)

run_button_pressed = st.button(
    "RUN Research",
    type="primary",
    disabled=st.session_state.research_running
)

# --- Core Logic ---
expected_sections = [
    "Company Overview", "Financial Highlights",
    "Recent News & Developments", "References & Citations"
]

if run_button_pressed and company_input:
    st.session_state.company_for_report = company_input
    st.session_state.research_running = True
    st.session_state.report_sections = {}

if st.session_state.research_running:
    st.header(f"Research Report for: {st.session_state.company_for_report}")
    st.markdown("---")

    is_initial_run = run_button_pressed and company_input

    if is_initial_run:
        # Execute research section by section using st.write_stream
        progress_bar = st.progress(0, text="Initializing research...")
        total_sections_count = len(expected_sections)

        try:
            for i, section_name in enumerate(expected_sections):
                st.subheader(section_name)
                progress_value = min(1.0, (i / total_sections_count))
                progress_bar.progress(progress_value, text=f"Processing: {section_name}...")

                section_content_generator = stream_section_data(
                    st.session_state.company_for_report,
                    section_name
                )
                with st.container(): # Using container helps isolate write_stream's layout effects
                     full_section_response = st.write_stream(section_content_generator)

                st.session_state.report_sections[section_name] = full_section_response
                time.sleep(0.2) # Short pause after section completes

            progress_bar.progress(1.0, text="Research complete!")
            st.success(f"Research for {st.session_state.company_for_report} finished.")

        except Exception as e:
            st.error(f"An error occurred during research for section '{section_name}': {e}")
            if 'progress_bar' in locals(): progress_bar.progress(1.0, text="Research failed.")
        finally:
            st.session_state.research_running = False
            st.rerun() # Rerun to update UI state (e.g., re-enable button)

    else:
        # Handle Interrupted State
        st.warning("Research was interrupted. Displaying completed sections.")
        for section_name in expected_sections:
            if section_name in st.session_state.report_sections:
                st.subheader(section_name)
                st.write(st.session_state.report_sections[section_name])


elif not st.session_state.research_running and st.session_state.company_for_report and st.session_state.report_sections:
    # Display the last completed report if idle
    st.header(f"Last Research Report for: {st.session_state.company_for_report}")
    st.markdown("---")
    for section_name in expected_sections:
        if section_name in st.session_state.report_sections:
            st.subheader(section_name)
            st.write(st.session_state.report_sections[section_name])

st.markdown("---")
st.caption("SmartAssist Research Tool v1.2")

```

**3. `pages/02_Run_History.py`**

```python
# pages/02_Run_History.py
import streamlit as st

st.set_page_config(page_title="Run History", page_icon="📜")
st.title("Run History")
st.markdown("This section will display the history of past research runs.")
st.info("Feature coming soon!")
```

**4. `pages/03_Settings.py`**

```python
# pages/03_Settings.py
import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️")
st.title("Settings")
st.markdown("Configure application settings here.")
st.info("Feature coming soon!")
```

**5. `pages/04_Global_Records.py`**

```python
# pages/04_Global_Records.py
import streamlit as st

st.set_page_config(page_title="Global Records", page_icon="🌍")
st.title("Global Records")
st.markdown("View global benchmarks or aggregated research data.")
st.info("Feature coming soon!")

```

**How to Set Up and Run:**

1.  Create a directory named `your_app_directory` (or any name you prefer).
2.  Inside `your_app_directory`, save the code for `SmartAssist.py` and `research_workflow.py`.
3.  Create a subdirectory named `pages` inside `your_app_directory`.
4.  Inside the `pages` subdirectory, save the code for `02_Run_History.py`, `03_Settings.py`, and `04_Global_Records.py`.
5.  Make sure you have the necessary libraries installed:
    ```bash
    pip install streamlit pandas numpy
    ```
6.  Open your terminal or command prompt.
7.  Navigate into `your_app_directory`:
    ```bash
    cd path/to/your_app_directory
    ```
8.  Run the Streamlit app:
    ```bash
    streamlit run SmartAssist.py
    ```

Your browser should open automatically, displaying the Streamlit application.